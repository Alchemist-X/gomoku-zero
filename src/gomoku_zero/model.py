"""Residual policy/WDL network, legal-action masking, and inference helpers."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .game import BOARD_CELLS, BOARD_SIZE, WDL_SIZE, Board

CHECKPOINT_SCHEMA_VERSION = 1


def masked_softmax(
    logits: torch.Tensor,
    legal_mask: torch.Tensor | np.ndarray,
    dim: int = -1,
) -> torch.Tensor:
    """Softmax over legal actions only.

    Masking happens *before* normalization.  Illegal entries are disconnected
    from ``logits`` by ``masked_fill``, then explicitly zeroed after softmax, so
    their probabilities and gradients are exactly zero.  Shapes must match
    exactly; broadcasting a mask across samples is intentionally rejected.
    Every slice along ``dim`` must contain at least one legal action.
    """

    if not isinstance(logits, torch.Tensor):
        raise TypeError("logits must be a torch.Tensor")
    if not logits.is_floating_point():
        raise TypeError("logits must have a floating-point dtype")
    if logits.ndim == 0:
        raise ValueError("logits must have at least one dimension")
    if not -logits.ndim <= dim < logits.ndim:
        raise IndexError(f"dim {dim} is out of range for a {logits.ndim}-D tensor")

    if not isinstance(legal_mask, torch.Tensor):
        legal_mask = torch.as_tensor(legal_mask)
    if legal_mask.dtype is not torch.bool:
        raise TypeError("legal_mask must have boolean dtype")
    if tuple(legal_mask.shape) != tuple(logits.shape):
        raise ValueError(
            f"legal_mask shape {tuple(legal_mask.shape)} must exactly match "
            f"logits shape {tuple(logits.shape)}"
        )
    legal_mask = legal_mask.to(device=logits.device)

    has_legal_action = legal_mask.any(dim=dim)
    if not bool(has_legal_action.all().item()):
        raise ValueError("every softmax slice must contain at least one legal action")

    # Inspect only legal logits.  NaN/Inf on illegal entries is safely discarded
    # by masked_fill and therefore cannot poison the normalization.
    if not bool(torch.isfinite(logits.masked_select(legal_mask)).all().item()):
        raise ValueError("legal logits must all be finite")

    masked_logits = logits.masked_fill(~legal_mask, -torch.inf)
    probabilities = torch.softmax(masked_logits, dim=dim)
    probabilities = probabilities.masked_fill(~legal_mask, 0.0)

    # The second normalization makes the exact public invariant explicit even
    # for low-precision dtypes.  Its denominator depends only on legal logits.
    normalizer = probabilities.sum(dim=dim, keepdim=True)
    probabilities = probabilities / normalizer
    return probabilities.masked_fill(~legal_mask, 0.0)


class ResidualBlock(nn.Module):
    """Two-convolution residual block used by :class:`GomokuNet`."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        if isinstance(channels, bool) or not isinstance(channels, int) or channels <= 0:
            raise ValueError("channels must be a positive integer")
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = inputs
        outputs = F.relu(self.bn1(self.conv1(inputs)), inplace=False)
        outputs = self.bn2(self.conv2(outputs))
        return F.relu(outputs + residual, inplace=False)


class GomokuNet(nn.Module):
    """AlphaZero-style residual network with policy and absolute WDL heads.

    Input is ``[batch, 3, 15, 15]`` using :meth:`Board.encode`.  The policy
    output is 225 raw row-major logits.  The WDL output is three raw logits in
    absolute order ``[black win, draw, white win]``.  Callers must apply
    :func:`masked_softmax` to the policy logits before sampling or search.
    """

    def __init__(self, channels: int = 128, residual_blocks: int = 10) -> None:
        super().__init__()
        if isinstance(channels, bool) or not isinstance(channels, int) or channels <= 0:
            raise ValueError("channels must be a positive integer")
        if (
            isinstance(residual_blocks, bool)
            or not isinstance(residual_blocks, int)
            or residual_blocks < 0
        ):
            raise ValueError("residual_blocks must be a non-negative integer")

        self.channels = channels
        self.residual_blocks = residual_blocks

        self.stem = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=False),
        )
        self.tower = nn.Sequential(*(ResidualBlock(channels) for _ in range(residual_blocks)))

        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=False),
            nn.Flatten(),
            nn.Linear(2 * BOARD_CELLS, BOARD_CELLS),
        )
        self.wdl_features = nn.Sequential(
            nn.Conv2d(channels, 3, kernel_size=1, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=False),
            nn.Flatten(),
        )
        self.wdl_head = nn.Sequential(
            nn.Linear(3 * BOARD_CELLS, 256),
            nn.ReLU(inplace=False),
            nn.Linear(256, WDL_SIZE),
        )

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> GomokuNet:
        """Build from a mapping containing ``channels`` and ``residual_blocks``."""

        return cls(
            channels=int(config.get("channels", 128)),
            residual_blocks=int(config.get("residual_blocks", 10)),
        )

    def model_config(self) -> dict[str, int]:
        return {"channels": self.channels, "residual_blocks": self.residual_blocks}

    def forward(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(states, torch.Tensor):
            raise TypeError("states must be a torch.Tensor")
        if states.ndim != 4 or tuple(states.shape[1:]) != (3, BOARD_SIZE, BOARD_SIZE):
            raise ValueError(
                f"states must have shape [batch, 3, {BOARD_SIZE}, {BOARD_SIZE}], "
                f"got {tuple(states.shape)}"
            )
        if not states.is_floating_point():
            raise TypeError("states must have a floating-point dtype")

        features = self.tower(self.stem(states))
        policy_logits = self.policy_head(features)
        wdl_logits = self.wdl_head(self.wdl_features(features))
        return policy_logits, wdl_logits

    def save_checkpoint(
        self,
        path: str | os.PathLike[str],
        *,
        optimizer: torch.optim.Optimizer | None = None,
        step: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> Path:
        return save_checkpoint(
            path,
            self,
            optimizer=optimizer,
            step=step,
            metadata=metadata,
        )

    @classmethod
    def from_checkpoint(
        cls,
        path: str | os.PathLike[str],
        *,
        map_location: str | torch.device = "cpu",
        strict: bool = True,
    ) -> GomokuNet:
        model, _ = load_checkpoint(path, map_location=map_location, strict=strict)
        return model.eval()


# Descriptive aliases retained for consumers that name the two-head architecture.
PolicyValueNet = GomokuNet
ResidualPolicyWDLNet = GomokuNet


def save_checkpoint(
    path: str | os.PathLike[str],
    model: GomokuNet,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    step: int | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Atomically save model state and optional optimizer/training metadata."""

    if not isinstance(model, GomokuNet):
        raise TypeError("model must be a GomokuNet")
    if step is not None and (isinstance(step, bool) or not isinstance(step, int) or step < 0):
        raise ValueError("step must be a non-negative integer or None")

    checkpoint_path = Path(path).expanduser()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model_config": model.model_config(),
        "model_state_dict": model.state_dict(),
        "step": step,
        "metadata": dict(metadata or {}),
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{checkpoint_path.name}.",
        suffix=".tmp",
        dir=checkpoint_path.parent,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, checkpoint_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return checkpoint_path


def load_checkpoint(
    path: str | os.PathLike[str],
    *,
    model: GomokuNet | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> tuple[GomokuNet, dict[str, Any]]:
    """Load a trusted checkpoint and return ``(model, checkpoint_payload)``."""

    checkpoint_path = Path(path).expanduser()
    try:
        payload = torch.load(checkpoint_path, map_location=map_location, weights_only=True)
    except TypeError:
        # Compatibility with older PyTorch builds, while the project pins 2.6+.
        payload = torch.load(checkpoint_path, map_location=map_location)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint payload must be a dictionary")
    if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported checkpoint schema {payload.get('schema_version')!r}; "
            f"expected {CHECKPOINT_SCHEMA_VERSION}"
        )
    config = payload.get("model_config")
    state_dict = payload.get("model_state_dict")
    if not isinstance(config, dict) or not isinstance(state_dict, dict):
        raise ValueError("checkpoint is missing model_config or model_state_dict")

    if model is None:
        model = GomokuNet.from_config(config)
    elif strict and model.model_config() != {
        "channels": int(config.get("channels", -1)),
        "residual_blocks": int(config.get("residual_blocks", -1)),
    }:
        raise ValueError(
            f"checkpoint model config {config} does not match target {model.model_config()}"
        )
    model.load_state_dict(state_dict, strict=strict)

    if optimizer is not None:
        optimizer_state = payload.get("optimizer_state_dict")
        if optimizer_state is None:
            raise ValueError("checkpoint does not contain optimizer state")
        optimizer.load_state_dict(optimizer_state)
    return model, payload


@dataclass(frozen=True, slots=True)
class Evaluation:
    """Detached CPU result from :class:`DeterministicEvaluator`."""

    policy: np.ndarray
    wdl: np.ndarray
    value: float
    policy_logits: np.ndarray | None = None
    wdl_logits: np.ndarray | None = None

    @property
    def priors(self) -> np.ndarray:
        return self.policy


def _read_only(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array).copy()
    array.setflags(write=False)
    return array


class DeterministicEvaluator:
    """Deterministic, legality-safe adapter from :class:`Board` to MCTS.

    :meth:`evaluate` returns the full absolute WDL distribution and a derived
    scalar value.  Calling the object directly returns ``(policy_priors,
    absolute_wdl)`` to match :mod:`gomoku_zero.mcts`' evaluator contract.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        device: str | torch.device | None = None,
    ) -> None:
        if not isinstance(model, nn.Module):
            raise TypeError("model must be a torch.nn.Module")
        self.model = model
        if device is None:
            try:
                device = next(model.parameters()).device
            except StopIteration:
                device = torch.device("cpu")
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

    def evaluate(self, board: Board) -> Evaluation:
        if not isinstance(board, Board):
            raise TypeError("board must be a Board")

        if board.is_terminal:
            wdl = board.wdl_target()
            absolute_value = float(wdl[0] - wdl[2])
            return Evaluation(
                policy=_read_only(np.zeros(BOARD_CELLS, dtype=np.float32)),
                wdl=_read_only(wdl),
                value=absolute_value * board.to_play,
            )

        states = board.to_tensor(device=self.device).unsqueeze(0)
        legal_mask = torch.as_tensor(
            board.legal_mask,
            dtype=torch.bool,
            device=self.device,
        ).unsqueeze(0)
        with torch.inference_mode():
            policy_logits, wdl_logits = self.model(states)
            if tuple(policy_logits.shape) != (1, BOARD_CELLS):
                raise ValueError(
                    f"model policy output must have shape (1, {BOARD_CELLS}), "
                    f"got {tuple(policy_logits.shape)}"
                )
            if tuple(wdl_logits.shape) != (1, WDL_SIZE):
                raise ValueError(
                    f"model WDL output must have shape (1, {WDL_SIZE}), "
                    f"got {tuple(wdl_logits.shape)}"
                )
            policy = masked_softmax(policy_logits, legal_mask, dim=-1)
            if not bool(torch.isfinite(wdl_logits).all().item()):
                raise ValueError("WDL logits must all be finite")
            wdl = torch.softmax(wdl_logits, dim=-1)

        policy_array = policy[0].detach().cpu().numpy().astype(np.float32, copy=False)
        wdl_array = wdl[0].detach().cpu().numpy().astype(np.float32, copy=False)
        policy_logits_array = policy_logits[0].detach().cpu().numpy()
        wdl_logits_array = wdl_logits[0].detach().cpu().numpy()
        absolute_value = float(wdl_array[0] - wdl_array[2])
        return Evaluation(
            policy=_read_only(policy_array),
            wdl=_read_only(wdl_array),
            value=absolute_value * board.to_play,
            policy_logits=_read_only(policy_logits_array),
            wdl_logits=_read_only(wdl_logits_array),
        )

    def __call__(self, board: Board) -> tuple[np.ndarray, np.ndarray]:
        result = self.evaluate(board)
        return result.policy.copy(), result.wdl.copy()


TorchEvaluator = DeterministicEvaluator


__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "DeterministicEvaluator",
    "Evaluation",
    "GomokuNet",
    "PolicyValueNet",
    "ResidualBlock",
    "ResidualPolicyWDLNet",
    "TorchEvaluator",
    "load_checkpoint",
    "masked_softmax",
    "save_checkpoint",
]
