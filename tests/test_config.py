from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest
import torch

from gomoku_zero.config import config_from_dict, load_config, seed_everything
from gomoku_zero.evaluation import evaluation_game_ids, wilson_interval

ROOT = Path(__file__).resolve().parents[1]


def test_production_config_has_formal_training_and_separate_evaluation() -> None:
    config = load_config(ROOT / "configs" / "production.json")

    assert config.training.total_self_play_games == 20_000
    assert config.training.self_play_actors == 28
    assert config.training.promotion_games == 200
    assert config.training.promotion_every == 10
    assert config.training.promotion_mcts_simulations == 200
    assert config.evaluation.games == 6_000
    assert config.evaluation.mcts_simulations == 800
    assert config.convergence.budgets == (128, 512, 2_048, 6_000, 12_000)
    assert config.serving.progressive_budgets[-1] == 6_000


def test_smoke_config_is_small() -> None:
    config = load_config(ROOT / "configs" / "smoke.json")

    assert config.training.total_self_play_games == 2
    assert config.training.self_play_actors == 1
    assert config.training.promotion_games == 0
    assert config.evaluation.games == 4
    assert config.convergence.budgets == (2, 4, 8)


def test_unknown_and_invalid_config_values_are_rejected() -> None:
    with pytest.raises(ValueError, match="unknown top-level"):
        config_from_dict({"seed": 1, "surprise": {}})
    with pytest.raises(ValueError, match="mcts_simulations"):
        config_from_dict({"seed": 1, "training": {"mcts_simulations": 0}})
    with pytest.raises(ValueError, match="unique and sorted"):
        config_from_dict(
            {"seed": 1, "convergence": {"budgets": [512, 128, 512]}}
        )


def test_seed_everything_is_reproducible_for_full_uint64_game_seeds() -> None:
    seed = 2**63 + 123_456_789
    first_rng = seed_everything(seed)
    first = (
        random.random(),
        np.random.random(),
        float(torch.rand(())),
        float(first_rng.random()),
    )
    second_rng = seed_everything(seed)
    second = (
        random.random(),
        np.random.random(),
        float(torch.rand(())),
        float(second_rng.random()),
    )

    assert first == second


def test_paired_evaluation_shards_keep_both_colors_together() -> None:
    shard_zero = evaluation_game_ids(
        8, shard_index=0, num_shards=2, paired_openings=True
    )
    shard_one = evaluation_game_ids(
        8, shard_index=1, num_shards=2, paired_openings=True
    )

    assert shard_zero == (0, 1, 4, 5)
    assert shard_one == (2, 3, 6, 7)
    assert wilson_interval(50, 100)[0] < 0.5 < wilson_interval(50, 100)[1]
