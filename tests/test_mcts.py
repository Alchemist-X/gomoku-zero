from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from gomoku_zero.game import BLACK, BOARD_CELLS, WHITE, Board
from gomoku_zero.mcts import (
    MCTSConfig,
    SearchSession,
    mask_and_normalize_policy,
    move_to_coordinate,
)


@dataclass(frozen=True, slots=True)
class ToyBoard:
    """Small rules stub expressed in the engine's fixed 225-action space."""

    occupied: frozenset[int] = frozenset()
    to_play: int = BLACK
    winning_move: int | None = None
    terminal_winner: int | None = None

    @property
    def winner(self) -> int | None:
        return self.terminal_winner

    @property
    def legal_mask(self) -> np.ndarray:
        mask = np.ones(BOARD_CELLS, dtype=np.bool_)
        if self.terminal_winner is not None:
            mask.fill(False)
        elif self.occupied:
            mask[np.fromiter(self.occupied, dtype=np.int64)] = False
        return mask

    @property
    def legal_moves(self) -> tuple[int, ...]:
        return tuple(int(move) for move in np.flatnonzero(self.legal_mask))

    def play(self, move: int) -> ToyBoard:
        if move not in self.legal_moves:
            raise ValueError(f"illegal move: {move}")
        winner = self.to_play if move == self.winning_move else None
        return ToyBoard(
            occupied=self.occupied | {move},
            to_play=WHITE if self.to_play == BLACK else BLACK,
            winning_move=self.winning_move,
            terminal_winner=winner,
        )


def neutral_evaluator(board: ToyBoard) -> tuple[np.ndarray, np.ndarray]:
    del board
    return np.ones(BOARD_CELLS), np.array([0.4, 0.2, 0.4])


def test_real_board_smoke_searches_only_legal_moves() -> None:
    board = Board.initial().play(112).play(0)
    result = SearchSession(board, neutral_evaluator).run(8)

    assert result.simulations == 8
    assert result.visit_counts.sum() == 8
    assert result.visit_counts[0] == 0
    assert result.visit_counts[112] == 0
    assert result.visit_policy[0] == 0.0
    assert result.visit_policy[112] == 0.0
    assert result.visit_policy.sum() == pytest.approx(1.0)


def test_policy_masking_renormalizes_only_legal_actions() -> None:
    mask = np.ones(BOARD_CELLS, dtype=np.bool_)
    mask[[0, 17, 224]] = False
    raw = np.zeros(BOARD_CELLS)
    raw[0] = 1_000_000.0
    raw[17] = np.inf
    raw[224] = np.nan

    policy = mask_and_normalize_policy(raw, mask)

    assert policy[[0, 17, 224]].tolist() == [0.0, 0.0, 0.0]
    assert policy.sum() == pytest.approx(1.0)
    assert np.all(policy[mask] == pytest.approx(1.0 / int(mask.sum())))


def test_adversarial_evaluator_cannot_create_or_visit_occupied_moves() -> None:
    occupied = frozenset({0, 1, 17, 112})
    board = ToyBoard(occupied=occupied)

    def adversarial_evaluator(current: ToyBoard) -> tuple[np.ndarray, np.ndarray]:
        policy = np.zeros(BOARD_CELLS)
        # Put all useful and malformed mass on points that are occupied in the
        # current position.  Search must fall back to legal-uniform priors.
        for move in current.occupied:
            policy[move] = 1_000_000.0
        policy[0] = np.inf
        policy[1] = np.nan
        return policy, np.array([0.5, 0.0, 0.5])

    result = SearchSession(board, adversarial_evaluator).run(64)
    expanded_moves = {item.move for item in result.moves}

    assert expanded_moves.isdisjoint(occupied)
    assert result.visit_counts[list(occupied)].sum() == 0
    assert result.visit_policy[list(occupied)].sum() == 0.0
    assert result.visit_policy.sum() == pytest.approx(1.0)
    assert result.visit_counts.sum() == 64
    assert all(item.network_prior > 0.0 for item in result.moves)


def test_terminal_win_is_exact_bypasses_evaluator_and_backs_up_root_sign() -> None:
    winning_move = 42
    board = ToyBoard(winning_move=winning_move)
    evaluated_terminal = False

    def evaluator(current: ToyBoard) -> tuple[np.ndarray, np.ndarray]:
        nonlocal evaluated_terminal
        if current.winner is not None:
            evaluated_terminal = True
            raise AssertionError("terminal boards must bypass the evaluator")
        policy = np.zeros(BOARD_CELLS)
        policy[winning_move] = 1.0
        # Deliberately claim that White wins; the exact child terminal must win.
        return policy, np.array([0.0, 0.0, 1.0])

    result = SearchSession(board, evaluator).run(12)
    winning_stats = next(item for item in result.moves if item.move == winning_move)

    assert not evaluated_terminal
    assert result.best_move == winning_move
    assert winning_stats.visits == 12
    assert winning_stats.q_value == pytest.approx(1.0)
    assert result.root_wdl == pytest.approx((1.0, 0.0, 0.0))
    assert result.root_value == pytest.approx(1.0)


def test_two_ply_terminal_value_alternates_back_to_root_perspective() -> None:
    setup_move = 10
    reply_win = 42
    board = ToyBoard(winning_move=reply_win)

    def evaluator(current: ToyBoard) -> tuple[np.ndarray, np.ndarray]:
        policy = np.zeros(BOARD_CELLS)
        policy[setup_move if not current.occupied else reply_win] = 1.0
        return policy, np.array([0.0, 1.0, 0.0])

    result = SearchSession(board, evaluator).run(2)
    setup_stats = next(item for item in result.moves if item.move == setup_move)

    # First visit evaluates the child as a draw; the second sees White's exact
    # reply win.  The edge mean from Black/root perspective is (0 + -1) / 2.
    assert setup_stats.visits == 2
    assert setup_stats.q_value == pytest.approx(-0.5)
    assert result.root_wdl == pytest.approx((0.0, 0.5, 0.5))
    assert result.root_value == pytest.approx(-0.5)


def test_progressive_budget_reuses_tree_and_counts_every_simulation_once() -> None:
    board = ToyBoard(occupied=frozenset({0, 1, 2}))
    session = SearchSession(board, neutral_evaluator)

    first = session.run_until(40)
    first_counts = first.visit_counts.copy()
    second = session.run_until(200)

    assert first.simulations == 40
    assert second.simulations == 200
    assert first_counts.sum() == 40
    assert second.visit_counts.sum() == 200
    assert np.all(second.visit_counts >= first_counts)

    # Repeating/lowering a target is idempotent; run() remains explicitly
    # additive for callers that already computed the delta.
    unchanged = session.run_until(100)
    assert unchanged.simulations == 200
    assert np.array_equal(unchanged.visit_counts, second.visit_counts)
    assert session.run(7).simulations == 207


def test_zero_simulation_snapshot_uses_masked_prior_and_deterministic_argmax() -> None:
    preferred = 91

    def evaluator(_: ToyBoard) -> tuple[np.ndarray, np.ndarray]:
        policy = np.zeros(BOARD_CELLS)
        policy[preferred] = 3.0
        policy[92] = 1.0
        return policy, np.array([0.3, 0.4, 0.3])

    result = SearchSession(ToyBoard(occupied=frozenset({0})), evaluator).result()

    assert result.simulations == 0
    assert result.best_move == preferred
    assert result.visit_policy.sum() == pytest.approx(1.0)
    assert result.visit_policy[preferred] == pytest.approx(0.75)
    deterministic = result.policy_at_temperature(0.0)
    assert deterministic[preferred] == 1.0
    assert np.count_nonzero(deterministic) == 1


def test_root_noise_is_seeded_normalized_and_only_applied_to_legal_children() -> None:
    board = ToyBoard(occupied=frozenset({0, 112, 224}))
    config = MCTSConfig(
        add_root_noise=True,
        dirichlet_alpha=0.2,
        dirichlet_epsilon=0.5,
        seed=1234,
    )

    left = SearchSession(board, neutral_evaluator, config).result()
    right = SearchSession(board, neutral_evaluator, config).result()

    left_priors = np.array([item.prior for item in left.moves])
    right_priors = np.array([item.prior for item in right.moves])
    assert np.array_equal(left_priors, right_priors)
    assert left_priors.sum() == pytest.approx(1.0)
    assert {item.move for item in left.moves}.isdisjoint(board.occupied)
    assert left.visit_policy[list(board.occupied)].sum() == 0.0


def test_full_board_is_exact_draw_and_never_calls_evaluator() -> None:
    board = ToyBoard(occupied=frozenset(range(BOARD_CELLS)))

    def forbidden_evaluator(_: ToyBoard) -> tuple[np.ndarray, np.ndarray]:
        raise AssertionError("full terminal boards must bypass the evaluator")

    result = SearchSession(board, forbidden_evaluator).run(9)

    assert result.simulations == 9
    assert result.moves == ()
    assert result.best_move is None
    assert result.visit_counts.sum() == 0
    assert result.visit_policy.sum() == 0.0
    assert result.root_wdl == pytest.approx((0.0, 1.0, 0.0))
    assert result.root_value == pytest.approx(0.0)


def test_one_legal_move_and_temperature_zero_remain_legal() -> None:
    only_move = 73
    occupied = frozenset(move for move in range(BOARD_CELLS) if move != only_move)
    result = SearchSession(ToyBoard(occupied=occupied), neutral_evaluator).run(5)

    assert result.visit_counts[only_move] == 5
    assert np.count_nonzero(result.visit_policy) == 1
    assert result.visit_policy[only_move] == 1.0
    assert np.array_equal(result.policy_at_temperature(0.0), result.visit_policy)


@pytest.mark.parametrize(
    ("move", "coordinate"),
    [(0, "A1"), (112, "H8"), (224, "O15")],
)
def test_coordinate_notation_includes_i(move: int, coordinate: str) -> None:
    assert move_to_coordinate(move) == coordinate
