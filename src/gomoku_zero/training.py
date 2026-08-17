"""Reproducible AlphaZero training loop and ``gomoku-train`` entry point."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from .config import RunConfig, TrainingConfig, load_config, seed_everything
from .model import CHECKPOINT_SCHEMA_VERSION, GomokuNet
from .replay import ReplayBuffer, TrainingSample
from .self_play import (
    SelfPlayGame,
    derive_game_seeds,
    generate_self_play_games,
    resolve_self_play_backend,
)

TRAINING_CHECKPOINT_VERSION = 1
PromotionHook = Callable[[torch.nn.Module, torch.nn.Module, int, int, int], Mapping[str, Any]]


def masked_log_softmax(
    logits: torch.Tensor,
    legal_mask: torch.Tensor,
) -> torch.Tensor:
    """Apply a per-position legality mask before log-softmax.

    Illegal log probabilities are set back to finite zero after normalization.
    This preserves exact zero gradients and avoids the ``0 * -inf`` NaN in a
    soft-target cross entropy.
    """

    if logits.ndim != 2 or logits.shape[1] != 225:
        raise ValueError("policy logits must have shape [batch, 225]")
    if legal_mask.dtype is not torch.bool or legal_mask.shape != logits.shape:
        raise ValueError("legal_mask must be bool and exactly match policy logits")
    if not bool(legal_mask.any(dim=1).all().item()):
        raise ValueError("each active policy row must contain a legal action")
    legal_logits = logits.masked_select(legal_mask)
    if not bool(torch.isfinite(legal_logits).all().item()):
        raise ValueError("legal policy logits must be finite")
    work = logits.float().masked_fill(~legal_mask, -torch.inf)
    return F.log_softmax(work, dim=-1).masked_fill(~legal_mask, 0.0)


def masked_policy_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    legal_mask: torch.Tensor,
    *,
    policy_valid: torch.Tensor | None = None,
    tolerance: float = 1e-6,
) -> torch.Tensor:
    """Soft policy cross-entropy whose denominator contains legal moves only."""

    if logits.ndim != 2 or logits.shape[1] != 225:
        raise ValueError("policy logits must have shape [batch, 225]")
    if target.shape != logits.shape or legal_mask.shape != logits.shape:
        raise ValueError("target and legal_mask must exactly match policy logits")
    if legal_mask.dtype is not torch.bool:
        raise TypeError("legal_mask must be boolean")
    if policy_valid is None:
        policy_valid = legal_mask.any(dim=1)
    if policy_valid.dtype is not torch.bool or policy_valid.shape != (logits.shape[0],):
        raise ValueError("policy_valid must be bool with shape [batch]")
    if not bool(torch.isfinite(target).all().item()) or bool((target < 0).any().item()):
        raise ValueError("policy targets must be finite and non-negative")
    if bool((target[~legal_mask].abs() > tolerance).any().item()):
        raise ValueError("policy target assigns mass to an illegal action")
    if bool((target[~policy_valid].abs() > tolerance).any().item()):
        raise ValueError("policy-invalid rows must have zero policy target")
    if not bool(policy_valid.any().item()):
        return logits.sum() * 0.0

    active_logits = logits[policy_valid]
    active_target = target[policy_valid].float()
    active_mask = legal_mask[policy_valid]
    sums = active_target.sum(dim=1)
    if not bool(torch.allclose(sums, torch.ones_like(sums), atol=tolerance, rtol=0.0)):
        raise ValueError("each active policy target must sum to one")
    log_probabilities = masked_log_softmax(active_logits, active_mask)
    return -(active_target * log_probabilities).sum(dim=1).mean()


def wdl_cross_entropy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 2 or logits.shape[1] != 3 or target.shape != logits.shape:
        raise ValueError("WDL logits and targets must have shape [batch, 3]")
    if not bool(torch.isfinite(target).all().item()) or bool((target < 0).any().item()):
        raise ValueError("WDL targets must be finite and non-negative")
    sums = target.float().sum(dim=1)
    if not bool(torch.allclose(sums, torch.ones_like(sums), atol=1e-6, rtol=0.0)):
        raise ValueError("each WDL target must sum to one")
    return -(target.float() * F.log_softmax(logits.float(), dim=-1)).sum(dim=1).mean()


@dataclass(frozen=True, slots=True)
class Losses:
    total: torch.Tensor
    policy: torch.Tensor
    wdl: torch.Tensor


def alpha_zero_loss(
    policy_logits: torch.Tensor,
    wdl_logits: torch.Tensor,
    policy_target: torch.Tensor,
    wdl_target: torch.Tensor,
    legal_mask: torch.Tensor,
) -> Losses:
    policy = masked_policy_cross_entropy(policy_logits, policy_target, legal_mask)
    wdl = wdl_cross_entropy(wdl_logits, wdl_target)
    return Losses(total=policy + wdl, policy=policy, wdl=wdl)


def _collate(
    samples: Sequence[TrainingSample], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    states = torch.as_tensor(
        np.stack([sample.encoded_state for sample in samples]),
        dtype=torch.float32,
        device=device,
    )
    policies = torch.as_tensor(
        np.stack([sample.policy_target for sample in samples]),
        dtype=torch.float32,
        device=device,
    )
    wdls = torch.as_tensor(
        np.stack([sample.wdl_target for sample in samples]),
        dtype=torch.float32,
        device=device,
    )
    masks = torch.as_tensor(
        np.stack([sample.legal_mask for sample in samples]),
        dtype=torch.bool,
        device=device,
    )
    return states, policies, wdls, masks


def train_steps(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    replay: ReplayBuffer,
    *,
    steps: int,
    batch_size: int,
    device: torch.device,
    gradient_clip_norm: float,
    mixed_precision: bool,
    scaler: torch.amp.GradScaler | None = None,
    symmetry: str = "random",
) -> dict[str, float]:
    if len(replay) == 0:
        raise ValueError("training requires at least one replay sample")
    if steps <= 0 or batch_size <= 0:
        raise ValueError("steps and batch_size must be positive")
    amp_enabled = mixed_precision and device.type == "cuda"
    model.train()
    totals = np.zeros(4, dtype=np.float64)
    for _ in range(steps):
        batch = replay.sample(batch_size, symmetry=symmetry)
        states, policies, wdls, masks = _collate(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            policy_logits, wdl_logits = model(states)
            losses = alpha_zero_loss(policy_logits, wdl_logits, policies, wdls, masks)
        if not torch.isfinite(losses.total):
            raise FloatingPointError("training loss became non-finite")
        if scaler is not None and amp_enabled:
            scaler.scale(losses.total).backward()
            scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), gradient_clip_norm, error_if_nonfinite=True
            )
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.total.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), gradient_clip_norm, error_if_nonfinite=True
            )
            optimizer.step()
        totals += (
            float(losses.total.detach()),
            float(losses.policy.detach()),
            float(losses.wdl.detach()),
            float(gradient_norm.detach()),
        )
    totals /= steps
    return {
        "loss": float(totals[0]),
        "policy_loss": float(totals[1]),
        "wdl_loss": float(totals[2]),
        "gradient_norm": float(totals[3]),
    }


def _cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _rng_state() -> dict[str, Any]:
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {
            "name": numpy_state[0],
            "keys": torch.from_numpy(numpy_state[1].astype(np.int64)),
            "position": numpy_state[2],
            "has_gauss": numpy_state[3],
            "cached_gaussian": numpy_state[4],
        },
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(tuple(state["python"]))
    numpy_state = state["numpy"]
    np.random.set_state(
        (
            str(numpy_state["name"]),
            numpy_state["keys"].cpu().numpy().astype(np.uint32),
            int(numpy_state["position"]),
            int(numpy_state["has_gauss"]),
            float(numpy_state["cached_gaussian"]),
        )
    )
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state.get("cuda"):
        torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda"]])


def _atomic_torch_save(payload: dict[str, Any], target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(payload: Mapping[str, Any], target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, target)


def _atomic_text(content: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _config_digest(config: RunConfig) -> str:
    encoded = json.dumps(config.to_dict(), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _model_digest(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _producer_revision() -> str:
    external = os.getenv("GIT_SHA") or os.getenv("COMMIT_SHA") or os.getenv("K_REVISION")
    if external:
        return external
    digest = hashlib.sha256()
    source_names = (
        "config.py",
        "game.py",
        "mcts.py",
        "model.py",
        "replay.py",
        "self_play.py",
        "training.py",
    )
    for name in source_names:
        path = Path(__file__).with_name(name)
        if path.exists():
            digest.update(name.encode())
            digest.update(path.read_bytes())
    return f"source-sha256:{digest.hexdigest()}"


def write_self_play_manifest(
    output_dir: str | os.PathLike[str],
    *,
    iteration: int,
    games: Sequence[SelfPlayGame],
    model: torch.nn.Module,
    config: RunConfig,
) -> Path:
    """Write an immutable, per-iteration provenance shard for self-play games."""

    path = (
        Path(output_dir).expanduser().resolve() / "self-play" / f"iteration-{iteration:04d}.jsonl"
    )
    common = {
        "schema_version": 1,
        "iteration": iteration,
        "producer_revision": _producer_revision(),
        "model_sha256": _model_digest(model),
        "config_sha256": _config_digest(config),
        "ruleset": "freestyle-v1",
    }
    lines = []
    for game_index, game in enumerate(games):
        row = {
            **common,
            "game_index": game_index,
            "seed": game.seed,
            "outcome": game.outcome,
            "moves": list(game.moves),
            "plies": len(game.moves),
            "samples": len(game.samples),
        }
        lines.append(json.dumps(row, sort_keys=True, separators=(",", ":")))
    content = "\n".join(lines) + ("\n" if lines else "")
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"immutable self-play manifest conflicts with {path}")
        return path
    _atomic_text(content, path)
    return path


def promotion_not_played(reason: str) -> dict[str, Any]:
    """Represent a disabled/skipped/non-scheduled gate without a false promotion."""

    return {"played": False, "promoted": None, "reason": reason}


def _restore_metrics_snapshot(snapshot: str, target: Path) -> None:
    """Restore missing checkpoint history without overwriting divergent history."""

    if not target.exists():
        _atomic_text(snapshot, target)
        return
    current = target.read_text(encoding="utf-8")
    if current == snapshot:
        return
    if snapshot.startswith(current):
        with target.open("a", encoding="utf-8") as handle:
            handle.write(snapshot[len(current) :])
            handle.flush()
            os.fsync(handle.fileno())
        return
    raise ValueError("existing metrics.jsonl diverges from checkpoint history")


def save_training_checkpoint(
    output_dir: str | os.PathLike[str],
    *,
    model: GomokuNet,
    champion_state_dict: Mapping[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler | None,
    replay: ReplayBuffer,
    config: RunConfig,
    iteration: int,
    global_step: int,
    skip_promotion: bool = False,
) -> Path:
    """Save an inference-compatible model checkpoint plus exact resume state.

    The large replay buffer is compressed into ``replay-latest.npz`` rather
    than duplicated into every versioned model checkpoint.  Only ``latest.pt``
    is guaranteed to be an exact replay-resume point; versioned ``iteration``
    files remain valid frozen models for evaluation and serving.
    """

    root = Path(output_dir).expanduser().resolve()
    checkpoint_dir = root / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    # Write a new versioned replay first.  Until latest.pt is replaced, the old
    # latest and its old replay remain a coherent recovery pair.
    replay_path = checkpoint_dir / f"replay-iteration-{iteration:04d}.npz"
    replay.save(replay_path)
    replay_digest = _sha256(replay_path)
    _atomic_json(
        {
            "iteration": iteration,
            "samples": len(replay),
            "bytes": replay_path.stat().st_size,
            "sha256": replay_digest,
        },
        replay_path.with_suffix(".json"),
    )

    metrics_path = root / "metrics.jsonl"
    metrics_snapshot = metrics_path.read_text(encoding="utf-8") if metrics_path.exists() else ""
    manifest_path = root / "self-play" / f"iteration-{iteration:04d}.jsonl"
    manifest_relative = None
    manifest_digest = None
    manifest_snapshot = None
    if manifest_path.exists():
        manifest_relative = str(manifest_path.relative_to(root))
        manifest_digest = _sha256(manifest_path)
        manifest_snapshot = manifest_path.read_text(encoding="utf-8")

    payload: dict[str, Any] = {
        # Preserve model.load_checkpoint compatibility for evaluation/serving.
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "training_checkpoint_version": TRAINING_CHECKPOINT_VERSION,
        "model_config": asdict(config.model),
        # Inference consumers always load the frozen promoted champion.  Exact
        # training resume additionally restores the in-flight learner.
        "model_state_dict": {
            name: tensor.detach().cpu().clone() for name, tensor in champion_state_dict.items()
        },
        "learner_state_dict": _cpu_state_dict(model),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "champion_state_dict": {
            name: tensor.detach().cpu().clone() for name, tensor in champion_state_dict.items()
        },
        "iteration": iteration,
        "step": global_step,
        "global_step": global_step,
        "run_config": config.to_dict(),
        "rng_state": _rng_state(),
        "replay_path": replay_path.name,
        "replay_iteration": iteration,
        "replay_sha256": replay_digest,
        "runtime_options": {"skip_promotion": skip_promotion},
        "metrics_jsonl": metrics_snapshot,
        "metrics_sha256": hashlib.sha256(metrics_snapshot.encode()).hexdigest(),
        "self_play_manifest": manifest_relative,
        "self_play_manifest_sha256": manifest_digest,
        "self_play_manifest_jsonl": manifest_snapshot,
        "metadata": {
            "kind": "gomoku-zero-training",
            "iteration": iteration,
            "config_sha256": _config_digest(config),
            "code_revision": _producer_revision(),
        },
    }
    versioned = checkpoint_dir / f"iteration-{iteration:04d}.pt"
    latest = checkpoint_dir / "latest.pt"
    _atomic_torch_save(payload, versioned)
    _atomic_torch_save(payload, latest)
    # Versioned .pt files are frozen model artifacts, not exact replay resume
    # points.  Once latest is durable, stale replay snapshots can be discarded.
    for stale in checkpoint_dir.glob("replay-iteration-*.npz"):
        if stale != replay_path:
            stale.unlink(missing_ok=True)
            stale.with_suffix(".json").unlink(missing_ok=True)
    return latest


def load_training_checkpoint(
    path: str | os.PathLike[str],
    *,
    model: GomokuNet,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler | None,
    replay: ReplayBuffer,
    expected_config: RunConfig,
    map_location: torch.device,
    expected_skip_promotion: bool = False,
    metrics_path: Path | None = None,
) -> tuple[int, int, dict[str, torch.Tensor]]:
    checkpoint_path = Path(path).expanduser().resolve()
    try:
        payload = torch.load(checkpoint_path, map_location=map_location, weights_only=True)
    except TypeError:  # pragma: no cover - project pins torch >= 2.6
        payload = torch.load(checkpoint_path, map_location=map_location)
    if payload.get("training_checkpoint_version") != TRAINING_CHECKPOINT_VERSION:
        raise ValueError("checkpoint is not a supported resumable training checkpoint")
    saved_model_config = payload.get("model_config")
    if saved_model_config != asdict(expected_config.model):
        raise ValueError("resume checkpoint model config does not match requested config")
    if int(payload.get("run_config", {}).get("seed", -1)) != expected_config.seed:
        raise ValueError("resume checkpoint seed does not match requested config")
    saved_runtime = payload.get("runtime_options", {})
    if bool(saved_runtime.get("skip_promotion", False)) != expected_skip_promotion:
        raise ValueError("resume must use the same --skip-promotion mode")
    saved_training = payload.get("run_config", {}).get("training", {})
    requested_training = asdict(expected_config.training)
    # Iteration count may be extended; all dynamics-affecting settings must match.
    saved_training = dict(saved_training)
    # Schema v1 checkpoints predate batched self-play.  Missing fields mean the
    # legacy process path and its inactive batching defaults.
    compatibility_defaults = TrainingConfig()
    for name in ("self_play_backend", "self_play_lanes", "inference_batch_size"):
        saved_training.setdefault(name, getattr(compatibility_defaults, name))
    saved_training.pop("iterations", None)
    requested_training.pop("iterations", None)
    if saved_training != requested_training:
        raise ValueError("resume checkpoint training hyperparameters do not match config")

    learner_state = payload.get("learner_state_dict", payload["model_state_dict"])
    model.load_state_dict(learner_state, strict=True)
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    scheduler.load_state_dict(payload["scheduler_state_dict"])
    if scaler is not None and payload.get("scaler_state_dict") is not None:
        scaler.load_state_dict(payload["scaler_state_dict"])
    replay_path = checkpoint_path.parent / payload["replay_path"]
    metadata_path = replay_path.with_suffix(".json")
    if not replay_path.exists() or not metadata_path.exists():
        raise FileNotFoundError("resume checkpoint is missing its replay artifact")
    replay_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if int(replay_metadata.get("iteration", -1)) != int(payload["replay_iteration"]):
        raise ValueError("replay artifact does not match checkpoint iteration")
    if int(replay_metadata.get("samples", -1)) < 0:
        raise ValueError("replay metadata is missing its sample count")
    if _sha256(replay_path) != payload.get("replay_sha256"):
        raise ValueError("replay artifact digest does not match checkpoint")
    replay.load(replay_path)
    if len(replay) != int(replay_metadata["samples"]):
        raise ValueError("replay artifact sample count does not match metadata")
    metrics_snapshot = str(payload.get("metrics_jsonl", ""))
    if hashlib.sha256(metrics_snapshot.encode()).hexdigest() != payload.get("metrics_sha256"):
        raise ValueError("checkpoint metrics snapshot digest mismatch")
    if metrics_path is not None:
        _restore_metrics_snapshot(metrics_snapshot, metrics_path)
        relative_manifest = payload.get("self_play_manifest")
        manifest_snapshot = payload.get("self_play_manifest_jsonl")
        if relative_manifest is not None and manifest_snapshot is not None:
            manifest_target = metrics_path.parent / str(relative_manifest)
            manifest_text = str(manifest_snapshot)
            if hashlib.sha256(manifest_text.encode()).hexdigest() != payload.get(
                "self_play_manifest_sha256"
            ):
                raise ValueError("checkpoint self-play manifest digest mismatch")
            if manifest_target.exists():
                if manifest_target.read_text(encoding="utf-8") != manifest_text:
                    raise ValueError("existing self-play manifest diverges from checkpoint")
            else:
                _atomic_text(manifest_text, manifest_target)
    _restore_rng_state(payload["rng_state"])
    champion = {
        name: tensor.detach().cpu().clone()
        for name, tensor in payload["champion_state_dict"].items()
    }
    return int(payload["iteration"]), int(payload["global_step"]), champion


def append_metric(path: Path, metric: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(metric), sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def estimate_training_work(config: RunConfig) -> dict[str, int | str]:
    games = config.training.total_self_play_games
    promotion_gates = config.training.iterations // config.training.promotion_every
    promotion_games = promotion_gates * config.training.promotion_games
    return {
        "self_play_games": games,
        "promotion_gates": promotion_gates,
        "promotion_evaluation_games": promotion_games,
        "promotion_serial_workers": 1,
        "promotion_mcts_simulations_per_move": config.training.promotion_mcts_simulations,
        "total_games": games + promotion_games,
        "actors": config.training.self_play_actors,
        "self_play_backend": config.training.self_play_backend,
        "self_play_lanes": config.training.self_play_lanes,
        "inference_batch_size": config.training.inference_batch_size,
        "mcts_simulations_per_move": config.training.mcts_simulations,
        "worst_case_leaf_evaluations": games * 225 * config.training.mcts_simulations,
        "promotion_worst_case_leaf_evaluations": promotion_games
        * 225
        * config.training.promotion_mcts_simulations,
        "optimizer_steps": config.training.iterations
        * config.training.training_steps_per_iteration,
    }


def _default_promotion_hook(
    config: RunConfig,
    *,
    device: torch.device,
) -> PromotionHook:
    def hook(
        candidate: torch.nn.Module,
        champion: torch.nn.Module,
        iteration: int,
        seed: int,
        games: int,
    ) -> Mapping[str, Any]:
        from .evaluation import evaluate_head_to_head

        return evaluate_head_to_head(
            candidate,
            champion,
            games=games,
            simulations=config.training.promotion_mcts_simulations,
            seed=seed,
            c_puct=config.training.c_puct,
            device=device,
        )

    return hook


def run_training(
    config: RunConfig,
    output_dir: str | os.PathLike[str],
    *,
    resume: str | os.PathLike[str] | None = None,
    initialize_from: str | os.PathLike[str] | None = None,
    device: str | torch.device = "cpu",
    actors: int | None = None,
    promotion_hook: PromotionHook | None = None,
    skip_promotion: bool = False,
) -> Path:
    if resume is not None and initialize_from is not None:
        raise ValueError("resume and initialize_from are mutually exclusive")
    device_value = torch.device(device)
    seed_everything(config.seed)
    metric_path = Path(output_dir).expanduser().resolve() / "metrics.jsonl"
    if resume is None and metric_path.exists():
        raise FileExistsError("metrics.jsonl already exists; pass --resume or use a new output dir")
    model = GomokuNet(
        channels=config.model.channels,
        residual_blocks=config.model.residual_blocks,
    ).to(device_value)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=list(config.training.learning_rate_milestones),
        gamma=config.training.learning_rate_gamma,
    )
    amp_enabled = config.training.mixed_precision and device_value.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    replay = ReplayBuffer(config.training.replay_buffer_size, seed=config.seed)
    champion_state = _cpu_state_dict(model)
    if initialize_from is not None:
        initialized = GomokuNet.from_checkpoint(
            initialize_from,
            map_location=device_value,
        ).to(device_value)
        if initialized.model_config() != model.model_config():
            raise ValueError("initial model architecture does not match requested config")
        model.load_state_dict(initialized.state_dict(), strict=True)
        champion_state = _cpu_state_dict(model)
    completed_iteration = 0
    global_step = 0
    if resume is not None:
        completed_iteration, global_step, champion_state = load_training_checkpoint(
            resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            replay=replay,
            expected_config=config,
            map_location=device_value,
            expected_skip_promotion=skip_promotion,
            metrics_path=metric_path,
        )
    if completed_iteration >= config.training.iterations:
        return Path(resume).expanduser().resolve() if resume else Path(output_dir).resolve()

    hook = promotion_hook or _default_promotion_hook(config, device=device_value)
    latest: Path | None = None
    last_checkpoint_iteration = 0
    for iteration in range(completed_iteration + 1, config.training.iterations + 1):
        started = time.monotonic()
        game_seeds = derive_game_seeds(
            config.seed, iteration, config.training.self_play_games_per_iteration
        )
        self_play_stats: dict[str, Any] = {}
        games = generate_self_play_games(
            model,
            config.model,
            config.training,
            game_seeds,
            actors=actors,
            device=device_value,
            stats=self_play_stats,
        )
        write_self_play_manifest(
            output_dir,
            iteration=iteration,
            games=games,
            model=model,
            config=config,
        )
        for game in games:
            replay.extend(game.samples)
        outcome_counts = {
            "black": sum(game.outcome == 1 for game in games),
            "draw": sum(game.outcome == 0 for game in games),
            "white": sum(game.outcome == -1 for game in games),
        }
        gate_enabled = (
            config.training.promotion_games > 0
            and iteration % config.training.promotion_every == 0
            and not skip_promotion
        )
        training_metrics = train_steps(
            model,
            optimizer,
            replay,
            steps=config.training.training_steps_per_iteration,
            batch_size=config.training.batch_size,
            device=device_value,
            gradient_clip_norm=config.training.gradient_clip_norm,
            mixed_precision=config.training.mixed_precision,
            scaler=scaler,
            symmetry=("random" if config.training.symmetry_augmentation == "random" else "none"),
        )
        global_step += config.training.training_steps_per_iteration

        if config.training.promotion_games == 0:
            promotion: dict[str, Any] = promotion_not_played("gating_disabled")
        elif skip_promotion:
            promotion = promotion_not_played("skipped_by_runtime_option")
        else:
            promotion = promotion_not_played("not_scheduled")
        if gate_enabled:
            champion = GomokuNet(
                channels=config.model.channels,
                residual_blocks=config.model.residual_blocks,
            ).to(device_value)
            champion.load_state_dict(champion_state)
            report = dict(
                hook(
                    model,
                    champion,
                    iteration,
                    int(game_seeds[0]),
                    config.training.promotion_games,
                )
            )
            score = float(report.get("candidate_score", math.nan))
            promoted = math.isfinite(score) and score >= config.training.promotion_threshold
            promotion = {**report, "played": True, "promoted": promoted}
            if promoted:
                champion_state = _cpu_state_dict(model)
            else:
                model.load_state_dict(champion_state, strict=True)
                # Candidate momentum is incompatible with restored champion
                # weights.  Reset moments explicitly; keep the global LR
                # schedule and AMP scale, which are iteration/numerics state.
                optimizer.state.clear()
        elif config.training.promotion_games == 0 or skip_promotion:
            champion_state = _cpu_state_dict(model)

        scheduler.step()
        metric: dict[str, Any] = {
            "event": "iteration",
            "iteration": iteration,
            "global_step": global_step,
            "games": len(games),
            "samples_generated": sum(len(game.samples) for game in games),
            "replay_samples": len(replay),
            "outcomes": outcome_counts,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "seconds": time.monotonic() - started,
            "self_play_actors": actors or config.training.self_play_actors,
            "self_play_inference": self_play_stats,
            "promotion": promotion,
            **training_metrics,
        }
        append_metric(metric_path, metric)
        if iteration % config.training.checkpoint_every == 0:
            latest = save_training_checkpoint(
                output_dir,
                model=model,
                champion_state_dict=champion_state,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                replay=replay,
                config=config,
                iteration=iteration,
                global_step=global_step,
                skip_promotion=skip_promotion,
            )
            last_checkpoint_iteration = iteration
    if last_checkpoint_iteration != config.training.iterations:
        latest = save_training_checkpoint(
            output_dir,
            model=model,
            champion_state_dict=champion_state,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            replay=replay,
            config=config,
            iteration=config.training.iterations,
            global_step=global_step,
            skip_promotion=skip_promotion,
        )
    return latest


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selected = torch.device(value)
    if selected.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    return selected


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="JSON run configuration")
    parser.add_argument("--output-dir", required=True, help="metrics/checkpoint directory")
    parser.add_argument("--resume", help="path to checkpoints/latest.pt")
    parser.add_argument(
        "--initialize-from",
        help="model checkpoint used only to initialize weights for a new run",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--actors", type=int, help="override self-play actor processes")
    parser.add_argument("--skip-promotion", action="store_true")
    parser.add_argument(
        "--allow-slow-production",
        action="store_true",
        help="acknowledge the cost of any run with at least 1,000 self-play games",
    )
    parser.add_argument(
        "--allow-batched-cpu",
        action="store_true",
        help="allow the GPU-oriented batched backend on CPU for benchmarks/tests",
    )
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    selected_device = _device(args.device)
    estimate = estimate_training_work(config)
    if args.actors is not None:
        if args.actors <= 0:
            raise SystemExit("--actors must be > 0")
        estimate["actors"] = args.actors
    resolved_backend = resolve_self_play_backend(config.training, selected_device)
    estimate["resolved_self_play_backend"] = resolved_backend
    estimate["selected_device"] = str(selected_device)
    estimate["batched_cpu_requires_override"] = (
        resolved_backend == "batched" and selected_device.type != "cuda"
    )
    if resolved_backend == "batched":
        estimate["maximum_effective_inference_batch"] = min(
            config.training.self_play_lanes,
            config.training.inference_batch_size,
            config.training.self_play_games_per_iteration,
        )
    else:
        estimate["maximum_effective_inference_batch"] = 1
    print(json.dumps({"event": "preflight", **estimate}, sort_keys=True), flush=True)
    if args.preflight_only:
        return 0
    if (
        resolved_backend == "batched"
        and selected_device.type != "cuda"
        and not args.allow_batched_cpu
    ):
        raise SystemExit(
            "batched self-play is GPU-oriented and requires --device cuda; "
            "pass --allow-batched-cpu only for an intentional CPU benchmark"
        )
    if resolved_backend == "batched" and args.actors is not None:
        raise SystemExit(
            "--actors applies only to the process backend; configure "
            "training.self_play_lanes for batched self-play"
        )
    is_large = estimate["self_play_games"] >= 1_000
    if is_large and not args.allow_slow_production:
        raise SystemExit(
            "large run refused: batching reduces inference calls but does not make the "
            "workload cheap; inspect the estimate and pass --allow-slow-production "
            "to acknowledge it"
        )
    latest = run_training(
        config,
        args.output_dir,
        resume=args.resume,
        initialize_from=args.initialize_from,
        device=selected_device,
        actors=args.actors,
        skip_promotion=args.skip_promotion,
    )
    print(json.dumps({"event": "complete", "checkpoint": str(latest)}), flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "Losses",
    "alpha_zero_loss",
    "estimate_training_work",
    "load_training_checkpoint",
    "main",
    "masked_log_softmax",
    "masked_policy_cross_entropy",
    "run_training",
    "save_training_checkpoint",
    "train_steps",
    "wdl_cross_entropy",
]
