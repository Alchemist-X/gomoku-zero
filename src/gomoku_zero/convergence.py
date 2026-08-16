"""Progressive MCTS search-budget convergence experiments.

One :class:`~gomoku_zero.mcts.SearchSession` is reused per position, so a
``128 -> 512 -> ...`` experiment performs only the additional simulations at
each rung. Reports contain absolute ``[black, draw, white]`` WDL estimates and
legal root visit shares at every configured budget.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import ConvergenceConfig, RunConfig, load_config, seed_everything
from .game import BLACK, BOARD_CELLS, BOARD_SIZE, WHITE, Board
from .mcts import Evaluator, MCTSConfig, MoveStats, SearchResult, SearchSession
from .model import DeterministicEvaluator, GomokuNet

_WDL_LABELS = ("black_win", "draw", "white_win")


def _move_report(move: MoveStats) -> dict[str, int | float | str]:
    return {
        "move": move.move,
        "coordinate": move.coordinate,
        "visits": move.visits,
        "visit_fraction": move.visit_fraction,
        "q_value": move.q_value,
        "network_prior": move.network_prior,
    }


def _top_moves(result: SearchResult, limit: int = 3) -> list[MoveStats]:
    """Rank legal root moves with the same deterministic tie-break as MCTS."""

    return sorted(
        result.moves,
        key=lambda move: (-move.visits, -move.q_value, move.move),
    )[:limit]


def _budget_snapshot(
    requested_budget: int,
    result: SearchResult,
    *,
    terminal_short_circuit: bool,
    incremental_elapsed_seconds: float,
    cumulative_elapsed_seconds: float,
) -> dict[str, Any]:
    top = _top_moves(result)
    return {
        "requested_budget": requested_budget,
        "simulations": result.simulations,
        "terminal_short_circuit": terminal_short_circuit,
        "incremental_elapsed_seconds": incremental_elapsed_seconds,
        "cumulative_elapsed_seconds": cumulative_elapsed_seconds,
        "top1": None if not top else _move_report(top[0]),
        "top3": [_move_report(move) for move in top],
        "absolute_wdl": dict(zip(_WDL_LABELS, result.root_wdl, strict=True)),
        "root_player": "black" if result.root_player == BLACK else "white",
        "root_value": result.root_value,
        "total_root_visits": int(result.visit_counts.sum()),
    }


def _comparison_pair(budgets: Sequence[int]) -> tuple[int, int]:
    if 6_000 in budgets and 12_000 in budgets:
        return 6_000, 12_000
    if len(budgets) < 2:
        raise ValueError("at least two convergence budgets are required for comparison")
    return int(budgets[-2]), int(budgets[-1])


def _comparison(
    snapshots: Mapping[int, Mapping[str, Any]],
    convergence: ConvergenceConfig,
) -> dict[str, Any]:
    lower_budget, upper_budget = _comparison_pair(convergence.budgets)
    lower = snapshots[lower_budget]
    upper = snapshots[upper_budget]

    lower_top3 = [int(move["move"]) for move in lower["top3"]]
    upper_top3 = [int(move["move"]) for move in upper["top3"]]
    lower_top1 = None if lower["top1"] is None else int(lower["top1"]["move"])
    upper_top1 = None if upper["top1"] is None else int(upper["top1"]["move"])
    same_top1 = lower_top1 == upper_top1
    same_top3_set = set(lower_top3) == set(upper_top3)

    lower_wdl = lower["absolute_wdl"]
    upper_wdl = upper["absolute_wdl"]
    deltas = {
        label: abs(float(upper_wdl[label]) - float(lower_wdl[label]))
        for label in _WDL_LABELS
    }
    max_delta = max(deltas.values())
    below_threshold = max_delta < convergence.max_win_rate_delta
    converged = below_threshold
    if convergence.require_same_top_move:
        converged = converged and same_top1
    if convergence.require_same_top_three_set:
        converged = converged and same_top3_set

    return {
        "lower_budget": lower_budget,
        "upper_budget": upper_budget,
        "same_top1": same_top1,
        "same_top3_set": same_top3_set,
        "wdl_absolute_deltas": deltas,
        "wdl_max_absolute_delta": max_delta,
        "wdl_threshold": convergence.max_win_rate_delta,
        "wdl_delta_below_threshold": below_threshold,
        "converged": converged,
    }


def run_position_convergence(
    board: Board,
    evaluator: Evaluator,
    convergence: ConvergenceConfig,
    *,
    c_puct: float = 1.5,
    seed: int = 0,
) -> dict[str, Any]:
    """Run one progressive convergence ladder and return a JSON-safe report.

    Terminal roots are evaluated exactly once by ``SearchSession`` and then
    short-circuited: every requested rung reports zero executed simulations,
    no legal recommendation, and the exact terminal WDL.
    """

    started = time.perf_counter()
    session = SearchSession(
        board,
        evaluator,
        MCTSConfig(c_puct=c_puct, add_root_noise=False, seed=seed),
    )
    setup_elapsed = time.perf_counter() - started
    snapshots: dict[int, dict[str, Any]] = {}
    if board.is_terminal:
        exact = session.result()
        for budget in convergence.budgets:
            snapshots[budget] = _budget_snapshot(
                budget,
                exact,
                terminal_short_circuit=True,
                incremental_elapsed_seconds=0.0,
                cumulative_elapsed_seconds=setup_elapsed,
            )
    else:
        for budget in convergence.budgets:
            rung_started = time.perf_counter()
            result = session.run_until(budget)
            rung_elapsed = time.perf_counter() - rung_started
            snapshots[budget] = _budget_snapshot(
                budget,
                result,
                terminal_short_circuit=False,
                incremental_elapsed_seconds=rung_elapsed,
                cumulative_elapsed_seconds=time.perf_counter() - started,
            )

    return {
        "budgets": [snapshots[budget] for budget in convergence.budgets],
        "comparison": _comparison(snapshots, convergence),
        "session_setup_seconds": setup_elapsed,
        "elapsed_seconds": time.perf_counter() - started,
    }


def _parse_player(value: object) -> int:
    if value in (BLACK, "black", "BLACK", "b", "B"):
        return BLACK
    if value in (WHITE, "white", "WHITE", "w", "W"):
        return WHITE
    raise ValueError("to_play must be 1/-1 or black/white")


def _coordinate_to_move(value: str) -> int:
    coordinate = value.strip().upper()
    if len(coordinate) < 2 or not "A" <= coordinate[0] <= "O":
        raise ValueError(f"invalid board coordinate: {value!r}")
    try:
        row = int(coordinate[1:]) - 1
    except ValueError as exc:
        raise ValueError(f"invalid board coordinate: {value!r}") from exc
    column = ord(coordinate[0]) - ord("A")
    if not 0 <= row < BOARD_SIZE:
        raise ValueError(f"invalid board coordinate: {value!r}")
    return row * BOARD_SIZE + column


def _parse_move(value: object) -> int:
    if isinstance(value, bool):
        raise ValueError("moves must be integer actions or A1-style coordinates")
    if isinstance(value, int):
        if 0 <= value < BOARD_CELLS:
            return value
        raise ValueError(f"move must be in [0, {BOARD_CELLS})")
    if isinstance(value, str):
        return _coordinate_to_move(value)
    raise ValueError("moves must be integer actions or A1-style coordinates")


def _board_from_position(raw: Mapping[str, Any]) -> Board:
    has_moves = "moves" in raw
    has_cells = "cells" in raw or "board" in raw
    if has_moves == has_cells:
        raise ValueError("each position must contain exactly one of moves or cells/board")

    if has_moves:
        moves = raw["moves"]
        if not isinstance(moves, list):
            raise ValueError("position.moves must be a JSON list")
        board = Board.empty()
        for ply, value in enumerate(moves):
            move = _parse_move(value)
            if not board.is_legal(move):
                raise ValueError(f"position contains illegal move at ply {ply}: {value!r}")
            board = board.play(move)
        if "to_play" in raw and _parse_player(raw["to_play"]) != board.to_play:
            raise ValueError("position.to_play conflicts with the replayed move sequence")
        return board

    cells = raw.get("cells", raw.get("board"))
    array = np.asarray(cells)
    if array.shape not in ((BOARD_CELLS,), (BOARD_SIZE, BOARD_SIZE)):
        raise ValueError("position cells must be a flat 225-list or 15x15 array")
    flat = array.reshape(-1)
    if "to_play" in raw:
        to_play = _parse_player(raw["to_play"])
    else:
        black_count = int(np.count_nonzero(flat == BLACK))
        white_count = int(np.count_nonzero(flat == WHITE))
        if black_count == white_count:
            to_play = BLACK
        elif black_count == white_count + 1:
            to_play = WHITE
        else:
            raise ValueError("non-alternating cells require an explicit position.to_play")
    # Omit last_move so Board performs a complete terminal-position scan.
    return Board.from_flat(flat, to_play=to_play)


def load_positions(path: str | os.PathLike[str]) -> list[tuple[str, Board]]:
    """Load one or more named analysis positions from a JSON document."""

    source = Path(path).expanduser()
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid positions JSON {source}: {exc}") from exc

    if isinstance(raw, Mapping) and "positions" in raw:
        items = raw["positions"]
    elif isinstance(raw, list):
        items = raw
    elif isinstance(raw, Mapping):
        items = [raw]
    else:
        raise ValueError("positions JSON must be an object, list, or {positions: [...]} object")
    if not isinstance(items, list) or not items:
        raise ValueError("positions JSON must contain at least one position")

    positions: list[tuple[str, Board]] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(items):
        if not isinstance(item, Mapping):
            raise ValueError(f"position {index} must be a JSON object")
        position_id = str(item.get("id", f"position-{index:04d}"))
        if not position_id or position_id in seen_ids:
            raise ValueError(f"position id must be non-empty and unique: {position_id!r}")
        seen_ids.add(position_id)
        positions.append((position_id, _board_from_position(item)))
    return positions


def _sha256(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _code_provenance() -> tuple[str, bool | None]:
    injected = (
        os.environ.get("GIT_SHA")
        or os.environ.get("COMMIT_SHA")
        or os.environ.get("K_REVISION")
        or os.environ.get("CODE_REVISION")
    )
    if injected:
        return injected, None
    repository = Path(__file__).resolve().parents[2]
    try:
        revision_result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            capture_output=True,
            check=False,
            text=True,
            timeout=2,
        )
        revision = revision_result.stdout.strip()
        if revision_result.returncode != 0 or not revision:
            return "unknown", None
        dirty_result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repository,
            capture_output=True,
            check=False,
            text=True,
            timeout=2,
        )
        dirty = None if dirty_result.returncode != 0 else bool(dirty_result.stdout.strip())
        return revision, dirty
    except (OSError, subprocess.SubprocessError):
        return "unknown", None


def _atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selected = torch.device(value)
    if selected.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    return selected


def run_convergence_experiment(
    config: RunConfig,
    checkpoint: str | os.PathLike[str],
    positions: Sequence[tuple[str, Board]],
    output: str | os.PathLike[str],
    *,
    device: str | torch.device = "cpu",
    config_source: str | os.PathLike[str] | None = None,
    positions_source: str | os.PathLike[str] | None = None,
) -> Path:
    """Run all supplied positions and atomically write a provenance-rich report."""

    if not positions:
        raise ValueError("at least one position is required")
    started = time.perf_counter()
    selected_device = torch.device(device)
    seed_everything(config.seed)
    model = GomokuNet.from_checkpoint(checkpoint, map_location=selected_device)
    evaluator = DeterministicEvaluator(model, device=selected_device)
    reports: list[dict[str, Any]] = []
    for index, (position_id, board) in enumerate(positions):
        position_report = run_position_convergence(
            board,
            evaluator,
            config.convergence,
            c_puct=config.training.c_puct,
            seed=config.seed + index,
        )
        reports.append(
            {
                "id": position_id,
                "move_count": board.move_count,
                "to_play": "black" if board.to_play == BLACK else "white",
                "status": board.status.value,
                "legal_actions": len(board.legal_moves),
                "cells": list(board.cells),
                **position_report,
            }
        )

    checkpoint_path = Path(checkpoint).expanduser().resolve()
    config_json = json.dumps(config.to_dict(), sort_keys=True, separators=(",", ":"))
    code_revision, code_dirty = _code_provenance()
    config_source_path = (
        None if config_source is None else Path(config_source).expanduser().resolve()
    )
    positions_source_path = (
        None if positions_source is None else Path(positions_source).expanduser().resolve()
    )
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "code_revision": code_revision,
        "code_dirty": code_dirty,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "config_source": None if config_source_path is None else str(config_source_path),
        "config_source_sha256": (
            None if config_source_path is None else _sha256(config_source_path)
        ),
        "config_sha256": hashlib.sha256(config_json.encode()).hexdigest(),
        "positions_source": (
            None if positions_source_path is None else str(positions_source_path)
        ),
        "positions_source_sha256": (
            None if positions_source_path is None else _sha256(positions_source_path)
        ),
        "seed": config.seed,
        "c_puct": config.training.c_puct,
        "device": str(selected_device),
        "model_config": model.model_config(),
        "absolute_wdl_order": list(_WDL_LABELS),
        "configured_budgets": list(config.convergence.budgets),
        "positions": reports,
    }
    output_path = Path(output).expanduser().resolve()
    _atomic_json(payload, output_path)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--positions", required=True, help="JSON file with one or more positions")
    parser.add_argument("--output", required=True, help="output JSON report")
    parser.add_argument("--device", default="auto")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    positions = load_positions(args.positions)
    output = run_convergence_experiment(
        config,
        args.checkpoint,
        positions,
        args.output,
        device=_device(args.device),
        config_source=args.config,
        positions_source=args.positions,
    )
    print(
        json.dumps(
            {
                "event": "complete",
                "positions": len(positions),
                "maximum_budget": config.convergence.budgets[-1],
                "output": str(output),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "build_parser",
    "load_positions",
    "main",
    "run_convergence_experiment",
    "run_position_convergence",
]
