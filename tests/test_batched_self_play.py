from __future__ import annotations

import json
from collections.abc import Sequence

import numpy as np
import pytest
import torch

from gomoku_zero.config import ModelConfig, TrainingConfig, config_from_dict
from gomoku_zero.game import BOARD_CELLS, Board
from gomoku_zero.mcts import MCTSConfig, SearchSession, search, search_many
from gomoku_zero.self_play import NeuralEvaluator, generate_self_play_games
from gomoku_zero.training import main as training_main


class RecordingBatchEvaluator:
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []
        self.evaluated_terminal = False

    def __call__(self, board: Board) -> tuple[np.ndarray, np.ndarray]:
        if board.is_terminal:
            self.evaluated_terminal = True
        policy = np.arange(1, BOARD_CELLS + 1, dtype=np.float64)
        return policy, np.array([0.45, 0.1, 0.45], dtype=np.float64)

    def evaluate_batch(self, boards: Sequence[Board]) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
        self.batch_sizes.append(len(boards))
        return tuple(self(board) for board in boards)


class ScriptedRecordingModel(torch.nn.Module):
    """Make Black fill row 1 and White fill row 2 for a nine-ply game."""

    def __init__(self) -> None:
        super().__init__()
        self.batch_sizes: list[int] = []

    def forward(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch = states.shape[0]
        self.batch_sizes.append(batch)
        logits = torch.full((batch, BOARD_CELLS), -100.0, device=states.device)
        black_to_move = states[:, 2, 0, 0] > 0.5
        for offset in range(5):
            logits[black_to_move, offset] = 10.0 - offset
            logits[~black_to_move, 15 + offset] = 10.0 - offset
        return logits, torch.zeros((batch, 3), device=states.device)


def _won_board() -> Board:
    board = Board.empty()
    for action in (0, 15, 1, 16, 2, 17, 3, 18, 4):
        board = board.play(action)
    assert board.is_terminal
    return board


def test_neural_evaluator_uses_one_batched_forward_and_masks_each_row() -> None:
    model = ScriptedRecordingModel()
    boards = (Board.empty(), Board.empty().play(0))
    evaluator = NeuralEvaluator(model)

    outputs = evaluator.evaluate_batch(boards)

    assert model.batch_sizes == [2]
    assert evaluator.telemetry()["mean_effective_batch_size"] == 2.0
    for board, (policy, wdl) in zip(boards, outputs, strict=True):
        assert policy[~board.legal_mask].sum() == 0.0
        assert policy[board.legal_mask].sum() == pytest.approx(1.0)
        assert wdl.sum() == pytest.approx(1.0)
    assert outputs[1][0][0] == 0.0


def test_search_many_batches_roots_and_leaves_without_changing_puct_results() -> None:
    boards = (
        Board.empty().play(0).play(15),
        Board.empty().play(112).play(0),
        Board.empty().play(224).play(1),
    )
    configs = tuple(MCTSConfig(seed=100 + index) for index in range(len(boards)))
    evaluator = RecordingBatchEvaluator()

    batched = search_many(boards, evaluator, 12, configs, max_batch_size=2)
    serial = tuple(
        search(board, RecordingBatchEvaluator(), 12, config)
        for board, config in zip(boards, configs, strict=True)
    )

    assert evaluator.batch_sizes[0] == 2  # root setup is batched too
    assert max(evaluator.batch_sizes) == 2
    for board, batched_result, serial_result in zip(boards, batched, serial, strict=True):
        assert batched_result.simulations == 12
        assert batched_result.visit_counts.sum() == 12
        assert np.array_equal(batched_result.visit_counts, serial_result.visit_counts)
        assert batched_result.root_wdl == pytest.approx(serial_result.root_wdl)
        assert batched_result.visit_policy[~board.legal_mask].sum() == 0.0


def test_batched_root_setup_is_not_a_simulation_and_terminal_bypasses_network() -> None:
    terminal = _won_board()
    active = Board.empty().play(112)
    evaluator = RecordingBatchEvaluator()

    zero = search_many((terminal, active), evaluator, 0, max_batch_size=4)
    searched = search_many((terminal, active), evaluator, 5, max_batch_size=4)

    assert [result.simulations for result in zero] == [0, 0]
    assert [result.simulations for result in searched] == [5, 5]
    assert searched[0].visit_counts.sum() == 0
    assert searched[1].visit_counts.sum() == 5
    assert not evaluator.evaluated_terminal
    # Only the active root is evaluated in each independent search_many call.
    assert evaluator.batch_sizes[0] == 1


def test_root_noise_is_applied_once_and_matches_batch_one_serial_semantics() -> None:
    board = Board.empty().play(112)
    config = MCTSConfig(
        seed=77,
        add_root_noise=True,
        dirichlet_alpha=0.15,
        dirichlet_epsilon=0.25,
    )
    evaluator = RecordingBatchEvaluator()

    serial = search(board, evaluator, 20, config)
    batched = search_many((board,), evaluator, 20, (config,), max_batch_size=4)[0]

    assert np.array_equal(serial.visit_counts, batched.visit_counts)
    assert [item.prior for item in serial.moves] == pytest.approx(
        [item.prior for item in batched.moves]
    )


def test_failed_leaf_evaluation_does_not_leave_serial_session_pending() -> None:
    calls = 0

    def flaky(_: Board) -> tuple[np.ndarray, np.ndarray]:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("temporary inference failure")
        return np.ones(BOARD_CELLS), np.array([0.4, 0.2, 0.4])

    session = SearchSession(Board.empty(), flaky)
    with pytest.raises(RuntimeError, match="temporary"):
        session.run(1)

    result = session.run(1)
    assert result.simulations == 1
    assert result.visit_counts.sum() == 1


def test_short_batch_evaluator_output_is_rejected() -> None:
    class ShortEvaluator(RecordingBatchEvaluator):
        def evaluate_batch(
            self, boards: Sequence[Board]
        ) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
            return super().evaluate_batch(boards)[:-1]

    with pytest.raises(ValueError, match="results for"):
        search_many((Board.empty(), Board.empty().play(0)), ShortEvaluator(), 1)


def test_public_batched_self_play_path_reports_real_batch_greater_than_one() -> None:
    model = ScriptedRecordingModel()
    config = TrainingConfig(
        iterations=1,
        self_play_games_per_iteration=3,
        self_play_actors=1,
        self_play_backend="batched",
        self_play_lanes=3,
        inference_batch_size=3,
        mcts_simulations=1,
        temperature_moves=0,
        training_steps_per_iteration=1,
        batch_size=2,
        replay_buffer_size=128,
        mixed_precision=False,
        promotion_games=0,
        symmetry_augmentation="none",
    )
    stats: dict[str, object] = {}

    games = generate_self_play_games(
        model,
        ModelConfig(channels=1, residual_blocks=1),
        config,
        (11, 12, 13),
        device="cpu",
        stats=stats,
    )
    repeated = generate_self_play_games(
        ScriptedRecordingModel(),
        ModelConfig(channels=1, residual_blocks=1),
        config,
        (11, 12, 13),
        device="cpu",
    )

    assert [game.seed for game in games] == [11, 12, 13]
    assert all(game.outcome == 1 for game in games)
    assert [game.moves for game in games] == [game.moves for game in repeated]
    for left, right in zip(games, repeated, strict=True):
        assert len(left.samples) == len(right.samples)
        for left_sample, right_sample in zip(left.samples, right.samples, strict=True):
            assert np.array_equal(left_sample.policy_target, right_sample.policy_target)
    assert max(model.batch_sizes) == 3
    assert stats["backend"] == "batched"
    assert stats["max_effective_batch_size"] == 3
    assert stats["mean_effective_batch_size"] == pytest.approx(3.0)
    assert int(stats["positions_evaluated"]) > 0


def test_batched_backend_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="self_play_backend"):
        config_from_dict({"seed": 1, "training": {"self_play_backend": "auto"}})
    with pytest.raises(ValueError, match="inference_batch_size"):
        config_from_dict({"seed": 1, "training": {"inference_batch_size": 0}})
    with pytest.raises(ValueError, match="self_play_lanes"):
        config_from_dict({"seed": 1, "training": {"self_play_lanes": 0}})


def test_gpu_profile_preflight_is_visible_on_cpu_but_execution_fails_fast(tmp_path, capsys) -> None:
    arguments = [
        "--config",
        "configs/gpu-batched.json",
        "--output-dir",
        str(tmp_path),
        "--device",
        "cpu",
    ]

    assert training_main([*arguments, "--preflight-only"]) == 0
    preflight = json.loads(capsys.readouterr().out)
    assert preflight["maximum_effective_inference_batch"] == 48
    assert preflight["batched_cpu_requires_override"] is True

    with pytest.raises(SystemExit, match="requires --device cuda"):
        training_main(arguments)
