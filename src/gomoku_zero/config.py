"""Validated, reproducible configuration for training and evaluation.

The JSON files in :mod:`configs` are deliberately plain so the same artifact can
be used locally, in a container, or by a managed training job.  This module is
the single place where defaults and validation live.
"""

from __future__ import annotations

import json
import os
import random
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _positive(name: str, value: int | float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value!r}")


def _probability(name: str, value: float, *, inclusive: bool = True) -> None:
    valid = 0.0 <= value <= 1.0 if inclusive else 0.0 < value < 1.0
    if not valid:
        bounds = "[0, 1]" if inclusive else "(0, 1)"
        raise ValueError(f"{name} must be in {bounds}, got {value!r}")


@dataclass(frozen=True, slots=True)
class ModelConfig:
    channels: int = 128
    residual_blocks: int = 10

    def __post_init__(self) -> None:
        _positive("model.channels", self.channels)
        _positive("model.residual_blocks", self.residual_blocks)


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    iterations: int = 100
    self_play_games_per_iteration: int = 200
    self_play_actors: int = 1
    self_play_backend: str = "process"
    self_play_lanes: int = 32
    inference_batch_size: int = 32
    mcts_simulations: int = 800
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.15
    dirichlet_fraction: float = 0.25
    temperature_moves: int = 16
    training_steps_per_iteration: int = 1_000
    batch_size: int = 256
    replay_buffer_size: int = 1_000_000
    learning_rate: float = 1e-3
    learning_rate_milestones: tuple[int, ...] = ()
    learning_rate_gamma: float = 0.2
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 5.0
    mixed_precision: bool = True
    checkpoint_every: int = 1
    promotion_games: int = 0
    promotion_every: int = 5
    promotion_mcts_simulations: int = 200
    promotion_threshold: float = 0.55
    symmetry_augmentation: str = "random"

    def __post_init__(self) -> None:
        for name in (
            "iterations",
            "self_play_games_per_iteration",
            "self_play_actors",
            "self_play_lanes",
            "inference_batch_size",
            "mcts_simulations",
            "training_steps_per_iteration",
            "batch_size",
            "replay_buffer_size",
            "checkpoint_every",
            "promotion_every",
            "promotion_mcts_simulations",
        ):
            _positive(f"training.{name}", getattr(self, name))
        for name in ("c_puct", "dirichlet_alpha", "learning_rate", "gradient_clip_norm"):
            _positive(f"training.{name}", getattr(self, name))
        if self.temperature_moves < 0:
            raise ValueError("training.temperature_moves must be >= 0")
        if self.weight_decay < 0:
            raise ValueError("training.weight_decay must be >= 0")
        if self.promotion_games < 0:
            raise ValueError("training.promotion_games must be >= 0")
        _probability("training.dirichlet_fraction", self.dirichlet_fraction)
        _probability("training.promotion_threshold", self.promotion_threshold)
        _probability("training.learning_rate_gamma", self.learning_rate_gamma, inclusive=False)
        if tuple(sorted(set(self.learning_rate_milestones))) != self.learning_rate_milestones:
            raise ValueError("training.learning_rate_milestones must be unique and sorted")
        if any(step <= 0 or step >= self.iterations for step in self.learning_rate_milestones):
            raise ValueError("learning-rate milestones must fall strictly inside the run")
        if self.symmetry_augmentation not in {"none", "random", "all"}:
            raise ValueError("training.symmetry_augmentation must be none, random, or all")
        if self.self_play_backend not in {"process", "batched"}:
            raise ValueError("training.self_play_backend must be process or batched")

    @property
    def total_self_play_games(self) -> int:
        return self.iterations * self.self_play_games_per_iteration


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    games: int = 6_000
    mcts_simulations: int = 800
    opening_plies: int = 4
    paired_openings: bool = True
    confidence_level: float = 0.95
    # The estimate is games * max_game_plies * simulations.  It is intentionally
    # conservative and catches accidental 6000-games-by-6000-simulations jobs.
    max_estimated_simulations: int = 1_500_000_000

    def __post_init__(self) -> None:
        _positive("evaluation.games", self.games)
        _positive("evaluation.mcts_simulations", self.mcts_simulations)
        if self.opening_plies < 0:
            raise ValueError("evaluation.opening_plies must be >= 0")
        _probability("evaluation.confidence_level", self.confidence_level, inclusive=False)
        _positive("evaluation.max_estimated_simulations", self.max_estimated_simulations)

    def estimated_simulations(self, games: int | None = None, max_plies: int = 225) -> int:
        return (self.games if games is None else games) * max_plies * self.mcts_simulations


@dataclass(frozen=True, slots=True)
class ConvergenceConfig:
    budgets: tuple[int, ...] = (128, 512, 2_048, 6_000, 12_000)
    max_win_rate_delta: float = 0.01
    require_same_top_move: bool = True
    require_same_top_three_set: bool = True

    def __post_init__(self) -> None:
        if not self.budgets or any(value <= 0 for value in self.budgets):
            raise ValueError("convergence.budgets must contain positive budgets")
        if tuple(sorted(set(self.budgets))) != self.budgets:
            raise ValueError("convergence.budgets must be unique and sorted")
        _probability("convergence.max_win_rate_delta", self.max_win_rate_delta)


@dataclass(frozen=True, slots=True)
class ServingConfig:
    progressive_budgets: tuple[int, ...] = (40, 200, 1_000, 3_000, 6_000)
    max_simulations: int = 6_000

    def __post_init__(self) -> None:
        if not self.progressive_budgets or any(value <= 0 for value in self.progressive_budgets):
            raise ValueError("serving.progressive_budgets must contain positive budgets")
        if tuple(sorted(set(self.progressive_budgets))) != self.progressive_budgets:
            raise ValueError("serving.progressive_budgets must be unique and sorted")
        if self.progressive_budgets[-1] > self.max_simulations:
            raise ValueError("progressive budget exceeds serving.max_simulations")


@dataclass(frozen=True, slots=True)
class RunConfig:
    seed: int
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    convergence: ConvergenceConfig = field(default_factory=ConvergenceConfig)
    serving: ServingConfig = field(default_factory=ServingConfig)

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ValueError("seed must be >= 0")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _section(raw: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = raw.get(name, {})
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return dict(value)


def _tuple_fields(values: dict[str, Any], *names: str) -> dict[str, Any]:
    for name in names:
        if name in values:
            values[name] = tuple(values[name])
    return values


def config_from_dict(raw: Mapping[str, Any]) -> RunConfig:
    """Build and validate a :class:`RunConfig` from decoded JSON."""

    known = {"seed", "model", "training", "evaluation", "convergence", "serving"}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"unknown top-level config keys: {sorted(unknown)}")
    if "seed" not in raw:
        raise ValueError("config is missing required key: seed")
    training = _tuple_fields(_section(raw, "training"), "learning_rate_milestones")
    convergence = _tuple_fields(_section(raw, "convergence"), "budgets")
    serving = _tuple_fields(_section(raw, "serving"), "progressive_budgets")
    return RunConfig(
        seed=int(raw["seed"]),
        model=ModelConfig(**_section(raw, "model")),
        training=TrainingConfig(**training),
        evaluation=EvaluationConfig(**_section(raw, "evaluation")),
        convergence=ConvergenceConfig(**convergence),
        serving=ServingConfig(**serving),
    )


def load_config(path: str | os.PathLike[str]) -> RunConfig:
    config_path = Path(path)
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON config {config_path}: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise ValueError("configuration root must be a JSON object")
    return config_from_dict(raw)


def seed_everything(seed: int, *, deterministic: bool = True) -> np.random.Generator:
    """Seed Python, NumPy, and PyTorch and return the run-local NumPy RNG.

    ``PYTHONHASHSEED`` only affects newly started interpreters, but setting it
    here makes child processes inherit the requested value.
    """

    if seed < 0:
        raise ValueError("seed must be >= 0")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    # NumPy's legacy module-level RNG accepts only uint32, while SeedSequence,
    # default_rng, Python, and PyTorch can retain the full per-game uint64 seed.
    np.random.seed(seed % (2**32))
    torch_seed = seed % (2**64)
    torch.manual_seed(torch_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(torch_seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    return np.random.default_rng(seed)


__all__ = [
    "ConvergenceConfig",
    "EvaluationConfig",
    "ModelConfig",
    "RunConfig",
    "ServingConfig",
    "TrainingConfig",
    "config_from_dict",
    "load_config",
    "seed_everything",
]
