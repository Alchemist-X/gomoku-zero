import json

import numpy as np

from gomoku_zero.config import ConvergenceConfig
from gomoku_zero.convergence import load_positions, main, run_position_convergence
from gomoku_zero.game import BLACK, BOARD_CELLS, Board
from gomoku_zero.model import GomokuNet


class OccupiedBiasedEvaluator:
    def __init__(self, occupied: int) -> None:
        self.occupied = occupied
        self.calls = 0

    def __call__(self, board: Board) -> tuple[np.ndarray, np.ndarray]:
        self.calls += 1
        policy = np.arange(1, BOARD_CELLS + 1, dtype=np.float64)
        policy[self.occupied] = 1e30
        return policy, np.array([0.55, 0.1, 0.35], dtype=np.float64)


def test_progressive_convergence_masks_occupied_move_and_compares_last_rungs() -> None:
    occupied = 7 * 15 + 7
    board = Board.empty().play(occupied)
    evaluator = OccupiedBiasedEvaluator(occupied)
    config = ConvergenceConfig(budgets=(2, 4, 8), max_win_rate_delta=0.01)

    report = run_position_convergence(board, evaluator, config, seed=11)

    assert [snapshot["simulations"] for snapshot in report["budgets"]] == [2, 4, 8]
    for snapshot in report["budgets"]:
        assert snapshot["total_root_visits"] == snapshot["requested_budget"]
        assert occupied not in [move["move"] for move in snapshot["top3"]]
        assert all(board.is_legal(move["move"]) for move in snapshot["top3"])
        assert sum(move["visit_fraction"] for move in snapshot["top3"]) <= 1.0
        assert set(snapshot["absolute_wdl"]) == {"black_win", "draw", "white_win"}
    assert report["comparison"]["lower_budget"] == 4
    assert report["comparison"]["upper_budget"] == 8
    # Reusing one session evaluates the root only once; fresh sessions would add
    # two more root calls across this three-rung experiment.
    assert evaluator.calls <= 1 + 8


def test_terminal_position_short_circuits_without_evaluator_call() -> None:
    moves = [0, 15, 1, 16, 2, 17, 3, 18, 4]
    board = Board.empty()
    for move in moves:
        board = board.play(move)

    def fail_if_called(_: Board) -> tuple[np.ndarray, np.ndarray]:
        raise AssertionError("terminal root must not invoke the evaluator")

    config = ConvergenceConfig(budgets=(1, 3))
    report = run_position_convergence(board, fail_if_called, config)

    for snapshot in report["budgets"]:
        assert snapshot["terminal_short_circuit"] is True
        assert snapshot["simulations"] == 0
        assert snapshot["top1"] is None
        assert snapshot["top3"] == []
        assert snapshot["absolute_wdl"] == {
            "black_win": 1.0,
            "draw": 0.0,
            "white_win": 0.0,
        }
    assert report["comparison"]["wdl_max_absolute_delta"] == 0.0


def test_load_positions_supports_coordinates_and_full_terminal_scan(tmp_path) -> None:
    cells = [0] * BOARD_CELLS
    cells[:5] = [BLACK] * 5
    source = tmp_path / "positions.json"
    source.write_text(
        json.dumps(
            {
                "positions": [
                    {"id": "coordinate-sequence", "moves": ["H8", "I8"]},
                    {"id": "imported-win", "cells": cells, "to_play": "white"},
                ]
            }
        ),
        encoding="utf-8",
    )

    positions = load_positions(source)

    assert positions[0][0] == "coordinate-sequence"
    assert positions[0][1].cells[7 * 15 + 7] == BLACK
    assert positions[1][1].is_terminal
    assert positions[1][1].winner == BLACK


def test_cli_writes_bounded_checkpoint_report(tmp_path) -> None:
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "seed": 3,
                "model": {"channels": 4, "residual_blocks": 1},
                "convergence": {"budgets": [1, 2]},
            }
        ),
        encoding="utf-8",
    )
    checkpoint = tmp_path / "model.pt"
    GomokuNet(channels=4, residual_blocks=1).save_checkpoint(checkpoint)
    positions = tmp_path / "positions.json"
    positions.write_text('{"id":"after-H8","moves":["H8"]}', encoding="utf-8")
    output = tmp_path / "report.json"

    assert (
        main(
            [
                "--config",
                str(config),
                "--checkpoint",
                str(checkpoint),
                "--positions",
                str(positions),
                "--output",
                str(output),
                "--device",
                "cpu",
            ]
        )
        == 0
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["schema_version"] == 1
    assert report["configured_budgets"] == [1, 2]
    assert report["positions"][0]["id"] == "after-H8"
    assert report["positions"][0]["budgets"][-1]["simulations"] == 2
