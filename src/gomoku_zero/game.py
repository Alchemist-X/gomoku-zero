"""Immutable game state and rules for 15x15 free-style Gomoku.

Actions are flat, row-major indices: ``action = row * BOARD_SIZE + column``.
Five or more contiguous stones in any straight direction wins.  There are no
Renju-style forbidden moves.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from numbers import Integral
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch

BOARD_SIZE = 15
BOARD_CELLS = BOARD_SIZE * BOARD_SIZE
WIN_LENGTH = 5

BLACK = 1
WHITE = -1
EMPTY = 0

WDL_BLACK = 0
WDL_DRAW = 1
WDL_WHITE = 2
WDL_SIZE = 3

_VALID_STONES = frozenset((BLACK, WHITE, EMPTY))
_DIRECTIONS = ((0, 1), (1, 0), (1, 1), (1, -1))


class IllegalMoveError(ValueError):
    """Raised when an action cannot legally be played on a board."""


class InvalidBoardError(ValueError):
    """Raised when a board representation has an invalid shape or value."""


class GameStatus(StrEnum):
    """Complete, non-overlapping set of game states."""

    ONGOING = "ongoing"
    DRAW = "draw"
    BLACK_WIN = "black_win"
    WHITE_WIN = "white_win"


def move_to_index(row: int, column: int) -> int:
    """Convert a zero-based ``(row, column)`` pair to a row-major action."""

    if isinstance(row, bool) or not isinstance(row, Integral):
        raise TypeError("row must be an integer")
    if isinstance(column, bool) or not isinstance(column, Integral):
        raise TypeError("column must be an integer")
    row = int(row)
    column = int(column)
    if not (0 <= row < BOARD_SIZE and 0 <= column < BOARD_SIZE):
        raise ValueError(f"coordinates must be in [0, {BOARD_SIZE})")
    return row * BOARD_SIZE + column


def index_to_move(action: int) -> tuple[int, int]:
    """Convert a flat row-major action to ``(row, column)``."""

    action = _normalize_action(action)
    return divmod(action, BOARD_SIZE)


def _normalize_action(action: int | tuple[int, int]) -> int:
    if isinstance(action, tuple):
        if len(action) != 2:
            raise TypeError("a coordinate action must contain exactly (row, column)")
        return move_to_index(action[0], action[1])
    if isinstance(action, bool) or not isinstance(action, Integral):
        raise TypeError("action must be an integer or a (row, column) tuple")
    action = int(action)
    if not 0 <= action < BOARD_CELLS:
        raise IllegalMoveError(f"action must be in [0, {BOARD_CELLS})")
    return action


def _normalize_cells(cells: Sequence[int] | np.ndarray | Iterable[int]) -> tuple[int, ...]:
    if isinstance(cells, np.ndarray):
        if cells.shape not in ((BOARD_CELLS,), (BOARD_SIZE, BOARD_SIZE)):
            raise InvalidBoardError(
                f"board array must have shape ({BOARD_CELLS},) or "
                f"({BOARD_SIZE}, {BOARD_SIZE}), got {cells.shape}"
            )
        values = cells.reshape(-1).tolist()
    else:
        try:
            values = list(cells)
        except TypeError as exc:
            raise InvalidBoardError("cells must be an iterable of board values") from exc
        if len(values) != BOARD_CELLS:
            raise InvalidBoardError(
                f"board must contain exactly {BOARD_CELLS} cells, got {len(values)}"
            )

    normalized: list[int] = []
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise InvalidBoardError(f"cell {index} must be an integer stone value")
        stone = int(value)
        if stone not in _VALID_STONES:
            raise InvalidBoardError(
                f"cell {index} has invalid value {stone}; expected BLACK, WHITE, or EMPTY"
            )
        normalized.append(stone)
    return tuple(normalized)


def absolute_wdl_target(outcome: int) -> np.ndarray:
    """Return the absolute ``[black win, draw, white win]`` one-hot target."""

    if isinstance(outcome, bool) or not isinstance(outcome, Integral):
        raise TypeError("outcome must be BLACK, EMPTY (draw), or WHITE")
    outcome = int(outcome)
    target = np.zeros(WDL_SIZE, dtype=np.float32)
    if outcome == BLACK:
        target[WDL_BLACK] = 1.0
    elif outcome == EMPTY:
        target[WDL_DRAW] = 1.0
    elif outcome == WHITE:
        target[WDL_WHITE] = 1.0
    else:
        raise ValueError("outcome must be BLACK, EMPTY (draw), or WHITE")
    return target


@dataclass(frozen=True, slots=True)
class Board:
    """An immutable, hashable Gomoku position.

    ``cells`` is normalized to an immutable tuple during construction.  Array
    accessors always allocate a copy, so a caller cannot mutate this position.
    The ordinary constructor intentionally permits analysis positions whose
    stone counts do not arise from alternating play; :meth:`play` itself always
    enforces legal sequential play.
    """

    cells: tuple[int, ...] = (EMPTY,) * BOARD_CELLS
    to_play: int = BLACK
    last_move: int | None = None
    _winning_line_cache: tuple[int, ...] | None = field(
        init=False,
        repr=False,
        compare=False,
        hash=False,
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "cells", _normalize_cells(self.cells))
        if isinstance(self.to_play, bool) or not isinstance(self.to_play, Integral):
            raise InvalidBoardError("to_play must be BLACK or WHITE")
        to_play = int(self.to_play)
        if to_play not in (BLACK, WHITE):
            raise InvalidBoardError("to_play must be BLACK or WHITE")
        object.__setattr__(self, "to_play", to_play)

        if self.last_move is not None:
            try:
                last_move = _normalize_action(self.last_move)
            except (TypeError, ValueError) as exc:
                raise InvalidBoardError("last_move is not a valid action") from exc
            if self.cells[last_move] == EMPTY:
                raise InvalidBoardError("last_move must refer to an occupied cell")
            object.__setattr__(self, "last_move", last_move)

        # Positions produced by play() cannot already contain an older win, so
        # only four lines through last_move can possibly create a new winner.
        # Imported analysis positions omit last_move and receive a full scan.
        if self.last_move is None:
            winning_line = self._scan_all_winning_lines()
        else:
            winning_line = self._winning_line_through(self.last_move)
        object.__setattr__(self, "_winning_line_cache", winning_line)

    @classmethod
    def empty(cls, *, to_play: int = BLACK) -> Board:
        """Create an empty position."""

        return cls(to_play=to_play)

    @classmethod
    def initial(cls) -> Board:
        """Create the standard initial position (black to move)."""

        return cls()

    @classmethod
    def from_flat(
        cls,
        cells: Sequence[int] | np.ndarray | Iterable[int],
        *,
        to_play: int = BLACK,
        last_move: int | None = None,
    ) -> Board:
        """Create a position from 225 row-major values."""

        return cls(cells=_normalize_cells(cells), to_play=to_play, last_move=last_move)

    @classmethod
    def from_array(
        cls,
        array: np.ndarray | Sequence[Sequence[int]],
        *,
        to_play: int = BLACK,
        last_move: int | None = None,
    ) -> Board:
        """Create a position from a 15x15 array-like object."""

        board = np.asarray(array)
        if board.shape != (BOARD_SIZE, BOARD_SIZE):
            raise InvalidBoardError(
                f"board array must have shape ({BOARD_SIZE}, {BOARD_SIZE}), got {board.shape}"
            )
        return cls(cells=_normalize_cells(board), to_play=to_play, last_move=last_move)

    @property
    def current_player(self) -> int:
        """Alias for :attr:`to_play`."""

        return self.to_play

    @property
    def move_count(self) -> int:
        return BOARD_CELLS - self.cells.count(EMPTY)

    @property
    def is_full(self) -> bool:
        return EMPTY not in self.cells

    def as_array(self) -> np.ndarray:
        """Return a writable 15x15 copy of the board."""

        return np.asarray(self.cells, dtype=np.int8).reshape(BOARD_SIZE, BOARD_SIZE).copy()

    @property
    def grid(self) -> np.ndarray:
        """A copy-safe array view for UI and serialization consumers."""

        return self.as_array()

    def at(self, row: int, column: int) -> int:
        return self.cells[move_to_index(row, column)]

    def __getitem__(self, action: int | tuple[int, int]) -> int:
        return self.cells[_normalize_action(action)]

    def _scan_all_winning_lines(self) -> tuple[int, ...] | None:
        for row in range(BOARD_SIZE):
            for column in range(BOARD_SIZE):
                stone = self.cells[row * BOARD_SIZE + column]
                if stone == EMPTY:
                    continue
                for delta_row, delta_column in _DIRECTIONS:
                    previous_row = row - delta_row
                    previous_column = column - delta_column
                    if (
                        0 <= previous_row < BOARD_SIZE
                        and 0 <= previous_column < BOARD_SIZE
                        and self.cells[previous_row * BOARD_SIZE + previous_column] == stone
                    ):
                        continue

                    line: list[int] = []
                    scan_row, scan_column = row, column
                    while (
                        0 <= scan_row < BOARD_SIZE
                        and 0 <= scan_column < BOARD_SIZE
                        and self.cells[scan_row * BOARD_SIZE + scan_column] == stone
                    ):
                        line.append(scan_row * BOARD_SIZE + scan_column)
                        scan_row += delta_row
                        scan_column += delta_column
                    if len(line) >= WIN_LENGTH:
                        return tuple(line)
        return None

    def _winning_line_through(self, action: int) -> tuple[int, ...] | None:
        stone = self.cells[action]
        if stone == EMPTY:
            return None
        row, column = divmod(action, BOARD_SIZE)
        for delta_row, delta_column in _DIRECTIONS:
            line = [action]

            scan_row = row - delta_row
            scan_column = column - delta_column
            before: list[int] = []
            while (
                0 <= scan_row < BOARD_SIZE
                and 0 <= scan_column < BOARD_SIZE
                and self.cells[scan_row * BOARD_SIZE + scan_column] == stone
            ):
                before.append(scan_row * BOARD_SIZE + scan_column)
                scan_row -= delta_row
                scan_column -= delta_column

            scan_row = row + delta_row
            scan_column = column + delta_column
            after: list[int] = []
            while (
                0 <= scan_row < BOARD_SIZE
                and 0 <= scan_column < BOARD_SIZE
                and self.cells[scan_row * BOARD_SIZE + scan_column] == stone
            ):
                after.append(scan_row * BOARD_SIZE + scan_column)
                scan_row += delta_row
                scan_column += delta_column

            line = list(reversed(before)) + line + after
            if len(line) >= WIN_LENGTH:
                return tuple(line)
        return None

    def winning_line(self) -> tuple[int, ...] | None:
        """Return the first maximal winning line, or ``None`` when ongoing/drawn."""

        return self._winning_line_cache

    @property
    def winner(self) -> int | None:
        line = self.winning_line()
        return None if line is None else self.cells[line[0]]

    @property
    def status(self) -> GameStatus:
        winner = self.winner
        if winner == BLACK:
            return GameStatus.BLACK_WIN
        if winner == WHITE:
            return GameStatus.WHITE_WIN
        if self.is_full:
            return GameStatus.DRAW
        return GameStatus.ONGOING

    @property
    def is_terminal(self) -> bool:
        return self.status is not GameStatus.ONGOING

    @property
    def terminal(self) -> bool:
        """Alias for :attr:`is_terminal`."""

        return self.is_terminal

    @property
    def is_draw(self) -> bool:
        return self.status is GameStatus.DRAW

    @property
    def outcome(self) -> int | None:
        """Return BLACK/WHITE winner, EMPTY for draw, or ``None`` if ongoing."""

        winner = self.winner
        if winner is not None:
            return winner
        return EMPTY if self.is_full else None

    @property
    def result(self) -> int | None:
        """Alias for :attr:`outcome`."""

        return self.outcome

    @property
    def legal_mask(self) -> np.ndarray:
        """Return a boolean mask over all 225 actions.

        Terminal positions have no legal continuation, even if a winning board
        still contains empty cells.
        """

        if self.is_terminal:
            return np.zeros(BOARD_CELLS, dtype=np.bool_)
        return np.fromiter((cell == EMPTY for cell in self.cells), dtype=np.bool_)

    @property
    def legal_moves(self) -> tuple[int, ...]:
        """All legal row-major actions, in ascending order."""

        if self.is_terminal:
            return ()
        return tuple(index for index, cell in enumerate(self.cells) if cell == EMPTY)

    def legal_actions(self) -> tuple[int, ...]:
        """Method-form alias for :attr:`legal_moves`."""

        return self.legal_moves

    def get_legal_mask(self) -> np.ndarray:
        """Method-form alias for :attr:`legal_mask`."""

        return self.legal_mask

    def get_legal_moves(self) -> tuple[int, ...]:
        """Method-form alias for :attr:`legal_moves`."""

        return self.legal_moves

    def is_legal(self, action: int | tuple[int, int]) -> bool:
        try:
            action_index = _normalize_action(action)
        except (TypeError, ValueError):
            return False
        return not self.is_terminal and self.cells[action_index] == EMPTY

    def play(self, action: int | tuple[int, int]) -> Board:
        """Return the position after ``action`` without mutating this one."""

        action_index = _normalize_action(action)
        if self.is_terminal:
            raise IllegalMoveError("cannot play a move after the game is terminal")
        if self.cells[action_index] != EMPTY:
            raise IllegalMoveError(f"action {action_index} is already occupied")
        next_cells = list(self.cells)
        next_cells[action_index] = self.to_play
        return Board(tuple(next_cells), to_play=-self.to_play, last_move=action_index)

    def apply_move(self, action: int | tuple[int, int]) -> Board:
        """Alias for :meth:`play`."""

        return self.play(action)

    def make_move(self, action: int | tuple[int, int]) -> Board:
        """Alias for :meth:`play`."""

        return self.play(action)

    def wdl_target(self) -> np.ndarray:
        """Return the terminal absolute WDL target ``[black, draw, white]``."""

        outcome = self.outcome
        if outcome is None:
            raise ValueError("an ongoing position has no final WDL target")
        return absolute_wdl_target(outcome)

    def encode(self) -> np.ndarray:
        """Encode as ``[black stones, white stones, black-to-move]`` planes.

        All planes are float32 and shaped ``[3, 15, 15]``.  The side-to-move
        plane is all ones for black and all zeros for white.
        """

        flat = np.asarray(self.cells, dtype=np.int8)
        black = (flat == BLACK).reshape(BOARD_SIZE, BOARD_SIZE)
        white = (flat == WHITE).reshape(BOARD_SIZE, BOARD_SIZE)
        side = np.full((BOARD_SIZE, BOARD_SIZE), self.to_play == BLACK, dtype=np.bool_)
        return np.stack((black, white, side)).astype(np.float32, copy=False)

    def to_tensor(self, *, device: str | torch.device | None = None) -> torch.Tensor:
        """Return :meth:`encode` as a PyTorch tensor (without a batch axis)."""

        import torch

        return torch.as_tensor(self.encode(), dtype=torch.float32, device=device)


def encode_board(board: Board) -> np.ndarray:
    """Functional alias for :meth:`Board.encode`."""

    if not isinstance(board, Board):
        raise TypeError("board must be a Board")
    return board.encode()


__all__ = [
    "BLACK",
    "BOARD_CELLS",
    "BOARD_SIZE",
    "EMPTY",
    "WHITE",
    "WIN_LENGTH",
    "WDL_BLACK",
    "WDL_DRAW",
    "WDL_SIZE",
    "WDL_WHITE",
    "Board",
    "GameStatus",
    "IllegalMoveError",
    "InvalidBoardError",
    "absolute_wdl_target",
    "encode_board",
    "index_to_move",
    "move_to_index",
]
