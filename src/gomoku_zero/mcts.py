"""PUCT Monte Carlo tree search with strict legal-action masking.

The evaluator contract is intentionally small: ``evaluator(board)`` returns a
225-element policy *probability* vector and an absolute ``[black, draw, white]``
WDL vector.  Policies are treated as untrusted input.  Occupied actions are
removed and the remaining mass is normalized before a node is expanded.

``SearchSession`` owns the tree and the random-number generator.  Calling
``run_until(40)`` and later ``run_until(200)`` therefore performs only 160 new
simulations and preserves all earlier statistics.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol, TypeAlias

import numpy as np
import numpy.typing as npt

from .game import BLACK, BOARD_CELLS, BOARD_SIZE, EMPTY, WHITE, Board

FloatArray: TypeAlias = npt.NDArray[np.float64]
Evaluator: TypeAlias = Callable[[Board], tuple[npt.ArrayLike, npt.ArrayLike]]


class BatchEvaluator(Protocol):
    """Evaluator that can execute one model forward for multiple positions."""

    def evaluate_batch(
        self, boards: Sequence[Board]
    ) -> Sequence[tuple[npt.ArrayLike, npt.ArrayLike]]: ...


@dataclass(frozen=True, slots=True)
class MCTSConfig:
    """Configuration for PUCT search.

    Root noise is disabled by default so analysis/evaluation is deterministic.
    Self-play should explicitly enable it.  A fixed seed makes noisy sessions
    reproducible as long as they receive the same sequence of calls.
    """

    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    add_root_noise: bool = False
    seed: int | None = 0

    def __post_init__(self) -> None:
        if not math.isfinite(self.c_puct) or self.c_puct < 0.0:
            raise ValueError("c_puct must be finite and non-negative")
        if not math.isfinite(self.dirichlet_alpha) or self.dirichlet_alpha <= 0.0:
            raise ValueError("dirichlet_alpha must be finite and positive")
        if not math.isfinite(self.dirichlet_epsilon) or not 0.0 <= self.dirichlet_epsilon <= 1.0:
            raise ValueError("dirichlet_epsilon must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class MoveStats:
    """Search statistics for one legal root move.

    ``q_value`` is always from the root side-to-move's perspective: ``+1`` is
    a root-player win and ``-1`` is a root-player loss.
    """

    move: int
    row: int
    col: int
    coordinate: str
    visits: int
    visit_fraction: float
    prior: float
    network_prior: float
    q_value: float

    def as_dict(self) -> dict[str, int | float | str]:
        return {
            "move": self.move,
            "row": self.row,
            "col": self.col,
            "coordinate": self.coordinate,
            "visits": self.visits,
            "visit_fraction": self.visit_fraction,
            "prior": self.prior,
            "network_prior": self.network_prior,
            "q_value": self.q_value,
        }


@dataclass(frozen=True, slots=True)
class SearchResult:
    """Immutable snapshot of a search session."""

    simulations: int
    root_player: int
    root_wdl: tuple[float, float, float]
    root_value: float
    visit_counts: npt.NDArray[np.int64]
    visit_policy: FloatArray
    moves: tuple[MoveStats, ...]
    best_move: int | None
    best_coordinate: str | None

    def policy_at_temperature(self, temperature: float) -> FloatArray:
        """Return a legal visit policy at ``temperature``.

        At temperature zero the result is a deterministic one-hot policy.  A
        zero-visit snapshot falls back to the already-masked root priors.
        """

        if not math.isfinite(temperature) or temperature < 0.0:
            raise ValueError("temperature must be finite and non-negative")

        policy = np.zeros(BOARD_CELLS, dtype=np.float64)
        if not self.moves:
            return policy

        moves = np.fromiter((item.move for item in self.moves), dtype=np.int64)
        counts = self.visit_counts[moves].astype(np.float64, copy=False)
        if not np.any(counts > 0.0):
            base = np.fromiter((item.prior for item in self.moves), dtype=np.float64)
        else:
            base = counts
        if temperature == 0.0:
            # ``moves`` is sorted, so np.argmax supplies a stable tie-break.
            policy[int(moves[int(np.argmax(base))])] = 1.0
            return policy

        positive = base > 0.0
        log_weights = np.full(base.shape, -np.inf, dtype=np.float64)
        log_weights[positive] = np.log(base[positive]) / temperature
        log_weights -= np.max(log_weights)
        weights = np.exp(log_weights)

        total = float(weights.sum())
        if not math.isfinite(total) or total <= 0.0:
            weights = np.ones(len(moves), dtype=np.float64)
            total = float(len(moves))
        policy[moves] = weights / total
        return policy

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-friendly representation."""

        return {
            "simulations": self.simulations,
            "root_player": self.root_player,
            "root_wdl": list(self.root_wdl),
            "root_value": self.root_value,
            "visit_counts": self.visit_counts.tolist(),
            "visit_policy": self.visit_policy.tolist(),
            "moves": [item.as_dict() for item in self.moves],
            "best_move": self.best_move,
            "best_coordinate": self.best_coordinate,
        }


@dataclass(slots=True)
class _Node:
    # Only selected edges receive a _Node and a materialized Board.  Legal moves
    # and priors stay in compact arrays, which keeps a 6000-simulation tree from
    # allocating roughly one Python object for every exposed empty point.
    board: Board
    move: int | None = None
    visit_count: int = 0
    value_sum: float = 0.0
    black_wdl_sum: float = 0.0
    draw_wdl_sum: float = 0.0
    white_wdl_sum: float = 0.0
    legal_moves: npt.NDArray[np.int16] | None = None
    priors: npt.NDArray[np.float32] | None = None
    # Needed only at the root, where callers can inspect pre-noise priors.
    network_priors: npt.NDArray[np.float32] | None = None
    children: dict[int, _Node] = field(default_factory=dict)
    expanded: bool = False
    terminal_wdl: FloatArray | None = None
    evaluation_wdl: FloatArray | None = None

    @property
    def q_value(self) -> float:
        return self.value_sum / self.visit_count if self.visit_count else 0.0


@dataclass(frozen=True, slots=True)
class _PendingSimulation:
    """One selected, non-terminal leaf awaiting network evaluation."""

    session: SearchSession
    node: _Node
    path: tuple[_Node, ...]
    legal_actions: tuple[int, ...]


def _property_value(obj: object, name: str) -> object:
    value = getattr(obj, name)
    return value() if callable(value) else value


def _legal_actions(board: Board) -> tuple[int, ...]:
    """Return the intersection of the board's two legal-action views.

    Consulting both ``legal_moves`` and ``legal_mask`` makes tree expansion
    fail closed if a custom Board implementation returns inconsistent data.
    """

    # The production Board is immutable and its cells are the source of truth.
    # _expand has already ruled out a winner, so reading cells directly avoids
    # repeating the relatively expensive winning-line scan in both public
    # legal accessors.  Duck-typed boards still take the defensive intersection
    # path below.
    if isinstance(board, Board):
        return tuple(index for index, stone in enumerate(board.cells) if stone == EMPTY)

    mask = np.asarray(_property_value(board, "legal_mask"), dtype=np.bool_).reshape(-1)
    if mask.size != BOARD_CELLS:
        raise ValueError(f"legal_mask must contain {BOARD_CELLS} entries, got {mask.size}")

    raw_moves = _property_value(board, "legal_moves")
    moves = {
        int(move)
        for move in raw_moves  # type: ignore[union-attr]
        if 0 <= int(move) < BOARD_CELLS and bool(mask[int(move)])
    }
    return tuple(sorted(moves))


def mask_and_normalize_policy(
    policy: npt.ArrayLike,
    legal_mask: npt.ArrayLike,
    legal_moves: Sequence[int] | None = None,
) -> FloatArray:
    """Zero illegal actions and renormalize over legal actions only.

    NaNs, infinities, and negative probabilities are treated as zero.  If the
    evaluator assigns no usable mass to legal actions, a legal uniform policy
    is returned.  A terminal/all-masked position returns all zeros.
    """

    values = np.asarray(policy, dtype=np.float64).reshape(-1)
    if values.size != BOARD_CELLS:
        raise ValueError(f"policy must contain {BOARD_CELLS} entries, got {values.size}")

    mask = np.asarray(legal_mask, dtype=np.bool_).reshape(-1)
    if mask.size != BOARD_CELLS:
        raise ValueError(f"legal_mask must contain {BOARD_CELLS} entries, got {mask.size}")
    if legal_moves is not None:
        move_mask = np.zeros(BOARD_CELLS, dtype=np.bool_)
        for move in legal_moves:
            index = int(move)
            if 0 <= index < BOARD_CELLS:
                move_mask[index] = True
        mask &= move_mask

    normalized = np.zeros(BOARD_CELLS, dtype=np.float64)
    legal = np.flatnonzero(mask)
    if legal.size == 0:
        return normalized

    legal_values = values[legal]
    legal_values = np.where(np.isfinite(legal_values) & (legal_values > 0.0), legal_values, 0.0)
    total = float(legal_values.sum())
    if not math.isfinite(total) or total <= 0.0:
        normalized[legal] = 1.0 / float(legal.size)
    else:
        normalized[legal] = legal_values / total
    return normalized


def _normalize_wdl(wdl: npt.ArrayLike) -> FloatArray:
    values = np.asarray(wdl, dtype=np.float64).reshape(-1)
    if values.size != 3:
        raise ValueError(f"WDL output must contain 3 entries, got {values.size}")
    values = np.where(np.isfinite(values) & (values >= 0.0), values, 0.0)
    total = float(values.sum())
    if not math.isfinite(total) or total <= 0.0:
        return np.full(3, 1.0 / 3.0, dtype=np.float64)
    return values / total


def _wdl_for_winner(winner: int | None) -> FloatArray | None:
    if winner == BLACK:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if winner == WHITE:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return None


def _value_for_player(wdl: npt.ArrayLike, player: int) -> float:
    values = np.asarray(wdl, dtype=np.float64)
    if player == BLACK:
        return float(values[0] - values[2])
    if player == WHITE:
        return float(values[2] - values[0])
    raise ValueError(f"unknown player value: {player!r}")


def move_to_coordinate(move: int) -> str:
    """Convert a zero-based flat action into an ``A1``-style coordinate."""

    if not 0 <= move < BOARD_CELLS:
        raise ValueError(f"move must be in [0, {BOARD_CELLS}), got {move}")
    row, col = divmod(move, BOARD_SIZE)
    return f"{chr(ord('A') + col)}{row + 1}"


class SearchSession:
    """A reusable PUCT tree rooted at one immutable board position."""

    def __init__(
        self,
        board: Board,
        evaluator: Evaluator,
        config: MCTSConfig | None = None,
        *,
        _defer_root_evaluation: bool = False,
    ) -> None:
        self.board = board
        self.evaluator = evaluator
        self.config = config or MCTSConfig()
        self._rng = np.random.default_rng(self.config.seed)
        self._root = _Node(board=board)
        self._simulations = 0
        self._pending: _PendingSimulation | None = None
        self._root_initialized = False

        # Root evaluation/expansion is setup, not a simulation.  Consequently
        # every requested simulation maps to exactly one root-child visit for a
        # non-terminal root.  ``search_many`` defers this work so all roots can
        # share the first network forward as well.
        if not _defer_root_evaluation:
            self._expand(self._root)
            if self.config.add_root_noise:
                self._add_root_dirichlet_noise()
            self._root_initialized = True

    @property
    def simulations(self) -> int:
        return self._simulations

    def run(self, additional_simulations: int) -> SearchResult:
        """Add exactly ``additional_simulations`` to the existing tree."""

        if isinstance(additional_simulations, bool) or not isinstance(
            additional_simulations, (int, np.integer)
        ):
            raise TypeError("additional_simulations must be an integer")
        if additional_simulations < 0:
            raise ValueError("additional_simulations must be non-negative")

        for _ in range(int(additional_simulations)):
            self._simulate()
        return self.result()

    def run_until(self, total_simulations: int) -> SearchResult:
        """Advance this session to at least ``total_simulations`` simulations."""

        if isinstance(total_simulations, bool) or not isinstance(
            total_simulations, (int, np.integer)
        ):
            raise TypeError("total_simulations must be an integer")
        if total_simulations < 0:
            raise ValueError("total_simulations must be non-negative")
        remaining = max(0, int(total_simulations) - self._simulations)
        return self.run(remaining)

    def result(self) -> SearchResult:
        """Take an immutable statistics snapshot without running a simulation."""

        if not self._root_initialized:
            raise RuntimeError("deferred search root has not been batch-initialized")

        visit_counts = np.zeros(BOARD_CELLS, dtype=np.int64)
        move_items: list[MoveStats] = []
        total_child_visits = sum(child.visit_count for child in self._root.children.values())
        root_moves = self._root.legal_moves
        root_priors = self._root.priors
        root_network_priors = self._root.network_priors
        if root_moves is None:
            root_moves = np.empty(0, dtype=np.int16)
        if root_priors is None:
            root_priors = np.empty(0, dtype=np.float32)
        if root_network_priors is None:
            root_network_priors = root_priors

        for index, raw_move in enumerate(root_moves):
            move = int(raw_move)
            child = self._root.children.get(move)
            visits = child.visit_count if child is not None else 0
            q_value = child.q_value if child is not None else 0.0
            row, col = divmod(move, BOARD_SIZE)
            visit_counts[move] = visits
            move_items.append(
                MoveStats(
                    move=move,
                    row=row,
                    col=col,
                    coordinate=move_to_coordinate(move),
                    visits=visits,
                    visit_fraction=(visits / total_child_visits if total_child_visits else 0.0),
                    prior=float(root_priors[index]),
                    network_prior=float(root_network_priors[index]),
                    q_value=q_value,
                )
            )

        if total_child_visits:
            visit_policy = visit_counts.astype(np.float64) / float(total_child_visits)
        elif move_items:
            visit_policy = np.zeros(BOARD_CELLS, dtype=np.float64)
            moves = np.fromiter((item.move for item in move_items), dtype=np.int64)
            priors = np.fromiter((item.prior for item in move_items), dtype=np.float64)
            prior_total = float(priors.sum())
            if prior_total <= 0.0 or not math.isfinite(prior_total):
                priors.fill(1.0 / float(len(priors)))
            else:
                priors /= prior_total
            visit_policy[moves] = priors
        else:
            visit_policy = np.zeros(BOARD_CELLS, dtype=np.float64)

        if move_items and total_child_visits:
            # Most visits, then highest Q, then smallest action for reproducible
            # deterministic evaluation when temperature is zero.
            best = min(move_items, key=lambda item: (-item.visits, -item.q_value, item.move))
        elif move_items:
            # Before any search, expose the network/noisy-prior recommendation.
            best = min(move_items, key=lambda item: (-item.prior, item.move))
        else:
            best = None

        if best is not None:
            best_move: int | None = best.move
            best_coordinate: str | None = best.coordinate
            best_child = self._root.children.get(best.move)
        else:
            best_move = None
            best_coordinate = None
            best_child = None

        # A searched root estimate follows the selected (most-visited) move,
        # rather than averaging leaf values from alternatives the search is
        # actively rejecting.  Before any edge visit, use the root network WDL;
        # terminal roots always carry their exact one-hot WDL.
        if best_child is not None and best_child.visit_count:
            root_wdl_array = np.array(
                [
                    best_child.black_wdl_sum,
                    best_child.draw_wdl_sum,
                    best_child.white_wdl_sum,
                ],
                dtype=np.float64,
            ) / float(best_child.visit_count)
        elif self._root.evaluation_wdl is not None:
            root_wdl_array = self._root.evaluation_wdl.copy()
        else:  # Defensive fallback; every expanded root has one of the above.
            root_wdl_array = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        root_wdl = tuple(float(item) for item in root_wdl_array)
        root_player = int(_property_value(self.board, "to_play"))
        root_value = _value_for_player(root_wdl_array, root_player)

        # Snapshots must not expose live mutable arrays.
        visit_counts.setflags(write=False)
        visit_policy.setflags(write=False)
        return SearchResult(
            simulations=self._simulations,
            root_player=root_player,
            root_wdl=(root_wdl[0], root_wdl[1], root_wdl[2]),
            root_value=root_value,
            visit_counts=visit_counts,
            visit_policy=visit_policy,
            moves=tuple(move_items),
            best_move=best_move,
            best_coordinate=best_coordinate,
        )

    @staticmethod
    def _terminal_or_legal(node: _Node) -> tuple[FloatArray | None, tuple[int, ...]]:
        board = node.board
        winner = _property_value(board, "winner")
        terminal_wdl = _wdl_for_winner(int(winner) if winner is not None else None)
        if terminal_wdl is not None:
            node.expanded = True
            node.terminal_wdl = terminal_wdl
            node.evaluation_wdl = terminal_wdl
            return terminal_wdl, ()

        legal_actions = _legal_actions(board)
        if not legal_actions:
            draw_wdl = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            node.expanded = True
            node.terminal_wdl = draw_wdl
            node.evaluation_wdl = draw_wdl
            return draw_wdl, ()
        return None, legal_actions

    def _expand(self, node: _Node) -> FloatArray:
        terminal_wdl, legal_actions = self._terminal_or_legal(node)
        if terminal_wdl is not None:
            return terminal_wdl

        raw_policy, raw_wdl = self._call_evaluator(node.board)
        return self._expand_evaluated(node, legal_actions, raw_policy, raw_wdl)

    def _expand_evaluated(
        self,
        node: _Node,
        legal_actions: Sequence[int],
        raw_policy: npt.ArrayLike,
        raw_wdl: npt.ArrayLike,
    ) -> FloatArray:
        if node.expanded:
            raise RuntimeError("cannot apply an evaluation to an expanded node")
        legal_mask = np.zeros(BOARD_CELLS, dtype=np.bool_)
        legal_mask[np.asarray(legal_actions, dtype=np.int64)] = True
        policy = mask_and_normalize_policy(
            raw_policy,
            legal_mask,
            legal_actions,
        )
        wdl = _normalize_wdl(raw_wdl)

        # Expansion stores only compact legal-action arrays.  A child Board and
        # node are constructed later, if and only if PUCT selects that action.
        node.legal_moves = np.asarray(legal_actions, dtype=np.int16)
        node.priors = policy[node.legal_moves].astype(np.float32, copy=True)
        if node is self._root:
            node.network_priors = node.priors.copy()
        node.expanded = True
        node.evaluation_wdl = wdl
        return wdl

    def _call_evaluator(self, board: Board) -> tuple[npt.ArrayLike, npt.ArrayLike]:
        """Evaluate while accepting the model adapter's richer result object.

        The canonical callable contract remains ``(policy, absolute_wdl)``.
        ``DeterministicEvaluator`` also exposes ``evaluate(board)`` returning an
        object with ``policy`` and ``wdl`` fields; preferring that path keeps the
        search directly compatible even with adapters whose legacy ``__call__``
        returns a scalar value.
        """

        evaluate = getattr(self.evaluator, "evaluate", None)
        if callable(evaluate):
            evaluated = evaluate(board)
            policy = getattr(evaluated, "policy", None)
            wdl = getattr(evaluated, "wdl", None)
            if policy is not None and wdl is not None:
                return policy, wdl
        return self.evaluator(board)

    def _add_root_dirichlet_noise(self) -> None:
        priors = self._root.priors
        network_priors = self._root.network_priors
        if priors is None or network_priors is None or priors.size == 0:
            return
        noise = self._rng.dirichlet(
            np.full(priors.size, self.config.dirichlet_alpha, dtype=np.float64)
        )
        epsilon = self.config.dirichlet_epsilon
        mixed = (1.0 - epsilon) * network_priors.astype(np.float64) + epsilon * noise
        total = float(mixed.sum())
        if total > 0.0 and math.isfinite(total):
            mixed /= total
        else:  # Defensive only; network priors and Dirichlet noise are valid.
            mixed.fill(1.0 / float(mixed.size))
        self._root.priors = mixed.astype(np.float32)

    def _select_child(self, node: _Node) -> _Node:
        legal_moves = node.legal_moves
        priors = node.priors
        if legal_moves is None or priors is None or legal_moves.size == 0:
            raise RuntimeError("cannot select a child from an unexpanded or terminal node")

        # Canonical PUCT uses the parent visit count.  max(1) lets the prior
        # choose the very first root edge even though root expansion is setup
        # and deliberately does not consume a simulation.
        parent_visits = max(1, node.visit_count)
        exploration_scale = self.config.c_puct * math.sqrt(float(parent_visits))

        best_move = -1
        best_score = -math.inf
        for index, raw_move in enumerate(legal_moves):
            move = int(raw_move)
            child = node.children.get(move)
            visits = child.visit_count if child is not None else 0
            q_value = child.q_value if child is not None else 0.0
            exploration = exploration_scale * float(priors[index]) / (1.0 + visits)
            score = q_value + exploration
            if score > best_score or (score == best_score and move < best_move):
                best_score = score
                best_move = move

        child = node.children.get(best_move)
        if child is None:
            child = _Node(board=node.board.play(best_move), move=best_move)
            node.children[best_move] = child
        return child

    def _prepare_simulation(self) -> _PendingSimulation | None:
        if not self._root_initialized:
            raise RuntimeError("deferred search root has not been batch-initialized")
        if self._pending is not None:
            raise RuntimeError("search session already has a pending leaf evaluation")
        node = self._root
        path: list[_Node] = []

        # There is no arbitrary depth cutoff.  Gomoku's acyclic tree guarantees
        # this walk is bounded by the number of remaining board cells.
        while node.expanded and node.terminal_wdl is None:
            child = self._select_child(node)
            path.append(child)
            node = child

        if node.expanded:
            if node.terminal_wdl is None:
                raise RuntimeError("expanded leaf has neither children nor terminal value")
            leaf_wdl = node.terminal_wdl
        else:
            terminal_wdl, legal_actions = self._terminal_or_legal(node)
            if terminal_wdl is None:
                self._pending = _PendingSimulation(self, node, tuple(path), legal_actions)
                return self._pending
            leaf_wdl = terminal_wdl

        self._finish_simulation(node, path, leaf_wdl)
        return None

    def _complete_simulation(
        self,
        pending: _PendingSimulation,
        raw_policy: npt.ArrayLike,
        raw_wdl: npt.ArrayLike,
    ) -> None:
        if pending.session is not self or self._pending is not pending:
            raise RuntimeError("pending simulation belongs to a different search session")
        leaf_wdl = self._expand_evaluated(
            pending.node,
            pending.legal_actions,
            raw_policy,
            raw_wdl,
        )
        self._finish_simulation(pending.node, pending.path, leaf_wdl)
        self._pending = None

    def _cancel_pending(self, pending: _PendingSimulation) -> None:
        if self._pending is pending:
            self._pending = None

    def _finish_simulation(
        self,
        node: _Node,
        path: Sequence[_Node],
        leaf_wdl: npt.ArrayLike,
    ) -> None:
        normalized_wdl = np.asarray(leaf_wdl, dtype=np.float64)

        # Evaluator WDL is absolute, so it can be accumulated directly for the
        # root estimate.  Scalar edge values alternate perspective at every ply.
        value = _value_for_player(normalized_wdl, int(_property_value(node.board, "to_play")))
        for child in reversed(path):
            value = -value
            child.visit_count += 1
            child.value_sum += value
            child.black_wdl_sum += float(normalized_wdl[0])
            child.draw_wdl_sum += float(normalized_wdl[1])
            child.white_wdl_sum += float(normalized_wdl[2])

        self._root.visit_count += 1
        self._root.value_sum += value
        self._simulations += 1

    def _simulate(self) -> None:
        pending = self._prepare_simulation()
        if pending is None:
            return
        try:
            raw_policy, raw_wdl = self._call_evaluator(pending.node.board)
            self._complete_simulation(pending, raw_policy, raw_wdl)
        except Exception:
            self._cancel_pending(pending)
            raise


def search(
    board: Board,
    evaluator: Evaluator,
    simulations: int,
    config: MCTSConfig | None = None,
) -> SearchResult:
    """Run a one-shot search; use :class:`SearchSession` for progressive work."""

    return SearchSession(board, evaluator, config).run(simulations)


def _batched_evaluate(
    evaluator: Evaluator,
    boards: Sequence[Board],
) -> tuple[tuple[npt.ArrayLike, npt.ArrayLike], ...]:
    evaluate_batch = getattr(evaluator, "evaluate_batch", None)
    if not callable(evaluate_batch):
        raise TypeError("batched search requires evaluator.evaluate_batch(boards)")
    raw_results = tuple(evaluate_batch(boards))
    if len(raw_results) != len(boards):
        raise ValueError(
            f"batch evaluator returned {len(raw_results)} results for {len(boards)} boards"
        )
    results: list[tuple[npt.ArrayLike, npt.ArrayLike]] = []
    for index, item in enumerate(raw_results):
        if not isinstance(item, Sequence) or len(item) != 2:
            raise ValueError(f"batch evaluator result {index} must be a (policy, WDL) pair")
        results.append((item[0], item[1]))
    return tuple(results)


def search_many(
    boards: Sequence[Board],
    evaluator: Evaluator,
    simulations: int,
    config: MCTSConfig | Sequence[MCTSConfig] | None = None,
    *,
    max_batch_size: int | None = None,
) -> tuple[SearchResult, ...]:
    """Search independent positions in lockstep with genuinely batched leaves.

    Each search tree has at most one outstanding leaf.  Selection therefore
    retains the serial PUCT semantics within a game and needs neither virtual
    loss nor locks; only independent leaves are coalesced.  Root evaluations
    are included in the batching so a move does not start with a burst of
    batch-one forwards.
    """

    if isinstance(simulations, bool) or not isinstance(simulations, (int, np.integer)):
        raise TypeError("simulations must be an integer")
    if simulations < 0:
        raise ValueError("simulations must be non-negative")
    board_items = tuple(boards)
    if not board_items:
        return ()
    if max_batch_size is None:
        batch_size = len(board_items)
    else:
        if isinstance(max_batch_size, bool) or not isinstance(max_batch_size, (int, np.integer)):
            raise TypeError("max_batch_size must be an integer")
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        batch_size = int(max_batch_size)

    if config is None or isinstance(config, MCTSConfig):
        configs = tuple((config or MCTSConfig()) for _ in board_items)
    else:
        configs = tuple(config)
        if len(configs) != len(board_items):
            raise ValueError("one MCTSConfig is required for every board")
        if not all(isinstance(item, MCTSConfig) for item in configs):
            raise TypeError("config sequence must contain only MCTSConfig values")

    sessions = tuple(
        SearchSession(
            board,
            evaluator,
            session_config,
            _defer_root_evaluation=True,
        )
        for board, session_config in zip(board_items, configs, strict=True)
    )

    # Root expansion is setup and must not increment simulation counts.
    root_requests: list[tuple[SearchSession, tuple[int, ...]]] = []
    for session in sessions:
        terminal_wdl, legal_actions = session._terminal_or_legal(session._root)
        if terminal_wdl is None:
            root_requests.append((session, legal_actions))
    for start in range(0, len(root_requests), batch_size):
        chunk = root_requests[start : start + batch_size]
        evaluations = _batched_evaluate(
            evaluator,
            tuple(session.board for session, _ in chunk),
        )
        for (session, legal_actions), (raw_policy, raw_wdl) in zip(chunk, evaluations, strict=True):
            session._expand_evaluated(
                session._root,
                legal_actions,
                raw_policy,
                raw_wdl,
            )
    for session in sessions:
        if session.config.add_root_noise:
            session._add_root_dirichlet_noise()
        session._root_initialized = True

    for _ in range(int(simulations)):
        pending: list[_PendingSimulation] = []
        for session in sessions:
            request = session._prepare_simulation()
            if request is not None:
                pending.append(request)
        try:
            for start in range(0, len(pending), batch_size):
                chunk = pending[start : start + batch_size]
                evaluations = _batched_evaluate(
                    evaluator,
                    tuple(request.node.board for request in chunk),
                )
                for request, (raw_policy, raw_wdl) in zip(chunk, evaluations, strict=True):
                    request.session._complete_simulation(request, raw_policy, raw_wdl)
        except Exception:
            for request in pending:
                request.session._cancel_pending(request)
            raise

    return tuple(session.result() for session in sessions)


__all__ = [
    "BatchEvaluator",
    "Evaluator",
    "MCTSConfig",
    "MoveStats",
    "SearchResult",
    "SearchSession",
    "mask_and_normalize_policy",
    "move_to_coordinate",
    "search",
    "search_many",
]
