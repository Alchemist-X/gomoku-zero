from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from gomoku_zero.game import (
    BLACK,
    BOARD_CELLS,
    BOARD_SIZE,
    EMPTY,
    WHITE,
    Board,
    GameStatus,
    IllegalMoveError,
    InvalidBoardError,
    index_to_move,
    move_to_index,
)


def _position(
    black: list[tuple[int, int]],
    white: list[tuple[int, int]] | None = None,
    *,
    to_play: int = BLACK,
) -> Board:
    cells = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
    for row, column in black:
        cells[row, column] = BLACK
    for row, column in white or []:
        cells[row, column] = WHITE
    return Board.from_array(cells, to_play=to_play)


def test_initial_board_and_row_major_coordinates() -> None:
    board = Board.initial()

    assert board.to_play == BLACK
    assert board.current_player == BLACK
    assert board.move_count == 0
    assert board.status == GameStatus.ONGOING
    assert not board.is_terminal
    assert board.winner is None
    assert board.outcome is None
    assert board.legal_moves == tuple(range(BOARD_CELLS))
    assert board.legal_mask.shape == (BOARD_CELLS,)
    assert board.legal_mask.dtype == np.bool_
    assert board.legal_mask.all()
    assert move_to_index(7, 7) == 112
    assert index_to_move(112) == (7, 7)


def test_play_is_immutable_and_array_access_is_copy_safe() -> None:
    board = Board()
    played = board.play((7, 7))

    assert board[112] == EMPTY
    assert played[112] == BLACK
    assert played.to_play == WHITE
    assert played.last_move == 112
    assert played.move_count == 1
    assert 112 not in played.legal_moves
    assert not played.legal_mask[112]

    array = played.as_array()
    array[7, 7] = EMPTY
    mask = played.legal_mask
    mask[0] = False
    assert played[112] == BLACK
    assert played.legal_mask[0]

    with pytest.raises(FrozenInstanceError):
        played.to_play = BLACK  # type: ignore[misc]


@pytest.mark.parametrize("action", [-1, BOARD_CELLS, True, 1.5, "0"])
def test_invalid_actions_are_strictly_rejected(action: object) -> None:
    with pytest.raises((IllegalMoveError, TypeError)):
        Board().play(action)  # type: ignore[arg-type]


def test_occupied_and_post_terminal_moves_are_rejected() -> None:
    board = Board().play(0)
    with pytest.raises(IllegalMoveError, match="occupied"):
        board.play(0)

    # Alternating filler moves let black win across the top row.
    board = Board()
    for move in (0, 15, 1, 16, 2, 17, 3, 18, 4):
        board = board.play(move)
    assert board.winner == BLACK
    assert board.is_terminal
    assert not board.legal_mask.any()
    assert board.legal_moves == ()
    with pytest.raises(IllegalMoveError, match="terminal"):
        board.play(19)


@pytest.mark.parametrize(
    "stones",
    [
        [(4, column) for column in range(3, 8)],
        [(row, 9) for row in range(5, 10)],
        [(row, row) for row in range(5)],
        [(row, 14 - row) for row in range(5)],
        [(10, column) for column in range(2, 8)],  # freestyle overline
    ],
)
def test_five_or_more_wins_in_all_directions(stones: list[tuple[int, int]]) -> None:
    board = _position(stones)

    assert board.winner == BLACK
    assert board.is_terminal
    assert board.status == GameStatus.BLACK_WIN
    assert len(board.winning_line() or ()) >= 5
    np.testing.assert_array_equal(board.wdl_target(), [1.0, 0.0, 0.0])


def test_gap_in_line_does_not_win() -> None:
    board = _position([(7, 3), (7, 4), (7, 6), (7, 7), (7, 8)])
    assert board.winner is None
    assert not board.is_terminal


def test_full_board_draw_has_no_legal_continuation() -> None:
    # This periodic pattern has runs of at most two in all four win directions.
    cells = np.empty((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
    for row in range(BOARD_SIZE):
        for column in range(BOARD_SIZE):
            cells[row, column] = (
                BLACK if (row + 2 * column) % 4 in {0, 1} else WHITE
            )
    board = Board.from_array(cells, to_play=WHITE)

    assert board.winner is None
    assert board.is_draw
    assert board.outcome == EMPTY
    assert board.status == GameStatus.DRAW
    assert board.legal_moves == ()
    assert not board.legal_mask.any()
    np.testing.assert_array_equal(board.wdl_target(), [0.0, 1.0, 0.0])


def test_tensor_encoding_has_black_white_and_side_planes() -> None:
    board = Board().play((2, 3)).play((8, 9))
    encoded = board.encode()

    assert encoded.shape == (3, BOARD_SIZE, BOARD_SIZE)
    assert encoded.dtype == np.float32
    assert encoded[0, 2, 3] == 1.0
    assert encoded[1, 8, 9] == 1.0
    assert encoded[0].sum() == 1.0
    assert encoded[1].sum() == 1.0
    assert encoded[2].min() == encoded[2].max() == 1.0  # black to play

    white_to_play = Board(to_play=WHITE).encode()
    assert not white_to_play[2].any()


def test_invalid_board_values_and_ongoing_wdl_are_rejected() -> None:
    with pytest.raises(InvalidBoardError):
        Board(cells=(EMPTY,) * (BOARD_CELLS - 1))
    with pytest.raises(InvalidBoardError):
        Board(cells=(2,) + (EMPTY,) * (BOARD_CELLS - 1))
    with pytest.raises(InvalidBoardError):
        Board(to_play=EMPTY)
    with pytest.raises(ValueError, match="ongoing"):
        Board().wdl_target()

