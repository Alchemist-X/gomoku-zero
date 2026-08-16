from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from gomoku_zero.config import load_config
from gomoku_zero.model import GomokuNet
from gomoku_zero.replay import (
    ReplayBuffer,
    TrainingSample,
    all_symmetries,
    normalize_policy_target,
    transform_sample,
)
from gomoku_zero.self_play import SelfPlayGame, sample_legal_action
from gomoku_zero.training import (
    _restore_metrics_snapshot,
    alpha_zero_loss,
    masked_log_softmax,
    masked_policy_cross_entropy,
    promotion_not_played,
    write_self_play_manifest,
)


def _sample_at(row: int = 2, col: int = 5) -> TrainingSample:
    state = np.zeros((3, 15, 15), dtype=np.float32)
    state[0, row, col] = 1.0
    legal = np.ones(225, dtype=np.bool_)
    legal[0] = False
    policy = np.zeros(225, dtype=np.float32)
    policy[row * 15 + col] = 1.0
    return TrainingSample(state, policy, np.array([1.0, 0.0, 0.0]), legal)


def test_masked_policy_loss_excludes_occupied_highest_logit_before_normalization() -> None:
    logits = torch.zeros((1, 225), dtype=torch.float64, requires_grad=True)
    with torch.no_grad():
        logits[0, 0] = 1_000.0  # occupied
        logits[0, 1] = 3.0
        logits[0, 2] = 2.5
    legal = torch.zeros((1, 225), dtype=torch.bool)
    legal[0, 1:3] = True
    target = torch.zeros((1, 225), dtype=torch.float64)
    target[0, 1] = 1.0

    log_probabilities = masked_log_softmax(logits, legal)
    probabilities = log_probabilities.exp().masked_fill(~legal, 0.0)
    loss = masked_policy_cross_entropy(logits, target, legal)
    loss.backward()

    assert probabilities[0, 0].item() == 0.0
    assert probabilities[0, 1].item() == pytest.approx(0.6224593312)
    assert probabilities[0, 2].item() == pytest.approx(0.3775406688)
    assert probabilities.sum().item() == pytest.approx(1.0)
    assert logits.grad is not None
    assert logits.grad[0, 0].item() == 0.0


def test_illegal_logit_cannot_change_loss_or_gradient() -> None:
    legal = torch.tensor([[False, True, True] + [False] * 222])
    target = torch.zeros((1, 225))
    target[0, 1] = 0.25
    target[0, 2] = 0.75
    values: list[float] = []
    for illegal_value in (-1e9, 1e9):
        logits = torch.zeros((1, 225), requires_grad=True)
        with torch.no_grad():
            logits[0, 0] = illegal_value
            logits[0, 1] = -2.0
            logits[0, 2] = 4.0
        loss = masked_policy_cross_entropy(logits, target, legal)
        loss.backward()
        values.append(float(loss.detach()))
        assert logits.grad is not None
        assert torch.count_nonzero(logits.grad[~legal]).item() == 0
    assert values[0] == pytest.approx(values[1])


def test_invalid_policy_targets_and_all_masked_active_rows_are_rejected() -> None:
    logits = torch.zeros((1, 225))
    legal = torch.ones((1, 225), dtype=torch.bool)
    legal[0, 0] = False
    illegal_target = torch.zeros((1, 225))
    illegal_target[0, 0] = 1.0
    with pytest.raises(ValueError, match="illegal"):
        masked_policy_cross_entropy(logits, illegal_target, legal)

    all_masked = torch.zeros((1, 225), dtype=torch.bool)
    with pytest.raises(ValueError, match="legal action"):
        masked_log_softmax(logits, all_masked)


def test_policy_invalid_terminal_batch_has_finite_differentiable_zero_loss() -> None:
    logits = torch.randn((2, 225), requires_grad=True)
    target = torch.zeros((2, 225))
    mask = torch.zeros((2, 225), dtype=torch.bool)
    loss = masked_policy_cross_entropy(
        logits,
        target,
        mask,
        policy_valid=torch.zeros(2, dtype=torch.bool),
    )
    loss.backward()

    assert loss.item() == 0.0
    assert logits.grad is not None
    assert torch.count_nonzero(logits.grad).item() == 0


def test_combined_loss_uses_absolute_three_way_wdl_target() -> None:
    policy_logits = torch.zeros((1, 225), requires_grad=True)
    wdl_logits = torch.tensor([[2.0, 1.0, -1.0]], requires_grad=True)
    policy_target = torch.full((1, 225), 1.0 / 225.0)
    wdl_target = torch.tensor([[1.0, 0.0, 0.0]])
    legal = torch.ones((1, 225), dtype=torch.bool)

    losses = alpha_zero_loss(
        policy_logits, wdl_logits, policy_target, wdl_target, legal
    )
    losses.total.backward()

    assert losses.total.item() == pytest.approx(losses.policy.item() + losses.wdl.item())
    assert wdl_logits.grad is not None
    assert wdl_logits.grad[0, 0].item() < 0.0


def test_policy_target_scrubs_tiny_residue_but_rejects_material_illegal_mass() -> None:
    legal = np.zeros(225, dtype=np.bool_)
    legal[1:3] = True
    weights = np.zeros(225, dtype=np.float64)
    weights[[0, 1, 2]] = [1e-10, 2.0, 1.0]
    normalized = normalize_policy_target(weights, legal)

    assert normalized[0] == 0.0
    assert normalized[1] == pytest.approx(2.0 / 3.0)
    assert normalized[2] == pytest.approx(1.0 / 3.0)
    with pytest.raises(ValueError, match="illegal actions"):
        weights[0] = 0.1
        normalize_policy_target(weights, legal)


def test_d4_symmetries_transform_state_policy_and_mask_together() -> None:
    sample = _sample_at(2, 5)
    expected = ((2, 5), (9, 2), (12, 9), (5, 12), (2, 9), (5, 2), (12, 5), (9, 12))

    augmented = all_symmetries(sample)
    assert len(augmented) == 8
    for symmetry, (transformed, (row, col)) in enumerate(zip(augmented, expected, strict=True)):
        target_index = row * 15 + col
        assert transformed.encoded_state[0, row, col] == 1.0
        assert transformed.policy_target[target_index] == 1.0
        assert transformed.policy_target[~transformed.legal_mask].sum() == 0.0
        assert transformed.policy_target.sum() == pytest.approx(1.0)
        assert np.array_equal(transformed.wdl_target, sample.wdl_target)
        assert np.array_equal(
            transformed.policy_target,
            transform_sample(sample, symmetry).policy_target,
        )


def test_replay_buffer_is_bounded_and_round_trips(tmp_path) -> None:
    replay = ReplayBuffer(2, seed=9)
    replay.extend([_sample_at(1, 1), _sample_at(2, 2), _sample_at(3, 3)])
    assert len(replay) == 2

    path = tmp_path / "replay.npz"
    replay.save(path)
    restored = ReplayBuffer(2, seed=0)
    restored.load(path)

    assert len(restored) == 2
    assert [int(np.argmax(item.policy_target)) for item in restored] == [32, 48]


def test_underfull_replay_resume_preserves_future_eviction_order(tmp_path) -> None:
    replay = ReplayBuffer(4, seed=1)
    replay.extend([_sample_at(0, 1), _sample_at(0, 2)])
    path = tmp_path / "underfull.npz"
    replay.save(path)
    restored = ReplayBuffer(4, seed=2)
    restored.load(path)
    restored.extend([_sample_at(0, 3), _sample_at(0, 4), _sample_at(0, 5)])

    assert [int(np.argmax(item.policy_target)) for item in restored] == [2, 3, 4, 5]


def test_stochastic_move_sampling_renormalizes_float32_policy_in_float64() -> None:
    legal = np.ones(225, dtype=np.bool_)
    policy = np.full(225, np.float32(1.0 / 225.0), dtype=np.float32)
    # The cast policy is intentionally not an exact float64 categorical sum.
    assert float(policy.astype(np.float64).sum()) != 1.0

    rng = np.random.default_rng(123)
    actions = [sample_legal_action(policy, legal, rng) for _ in range(100)]

    assert all(0 <= action < 225 for action in actions)


def test_metrics_resume_restores_tail_but_never_overwrites_divergent_history(
    tmp_path,
) -> None:
    metrics = tmp_path / "metrics.jsonl"
    metrics.write_text('{"iteration":1}\n', encoding="utf-8")
    snapshot = '{"iteration":1}\n{"iteration":2}\n'

    _restore_metrics_snapshot(snapshot, metrics)
    assert metrics.read_text(encoding="utf-8") == snapshot
    metrics.write_text(snapshot + '{"foreign":true}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="diverges"):
        _restore_metrics_snapshot(snapshot, metrics)
    assert metrics.read_text(encoding="utf-8").endswith('{"foreign":true}\n')


def test_self_play_manifest_is_traceable_and_immutable(tmp_path) -> None:
    config = load_config("configs/smoke.json")
    model = GomokuNet(channels=4, residual_blocks=0)
    game = SelfPlayGame(seed=123, samples=(), outcome=1, moves=(112, 113))

    path = write_self_play_manifest(
        tmp_path,
        iteration=1,
        games=[game],
        model=model,
        config=config,
    )
    row = json.loads(path.read_text(encoding="utf-8"))
    assert row["seed"] == 123
    assert row["outcome"] == 1
    assert row["moves"] == [112, 113]
    assert row["producer_revision"]
    assert len(row["model_sha256"]) == 64
    assert len(row["config_sha256"]) == 64

    write_self_play_manifest(
        tmp_path,
        iteration=1,
        games=[game],
        model=model,
        config=config,
    )
    conflict = SelfPlayGame(seed=123, samples=(), outcome=-1, moves=(112, 113))
    with pytest.raises(FileExistsError, match="immutable"):
        write_self_play_manifest(
            tmp_path,
            iteration=1,
            games=[conflict],
            model=model,
            config=config,
        )


def test_non_gate_promotion_status_is_not_a_promotion() -> None:
    status = promotion_not_played("not_scheduled")

    assert status == {
        "played": False,
        "promoted": None,
        "reason": "not_scheduled",
    }
