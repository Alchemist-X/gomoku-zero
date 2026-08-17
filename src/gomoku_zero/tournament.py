"""Batched round-robin checkpoint tournament with color-balanced Elo reporting."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import tempfile
import time
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .evaluation import generate_fixed_opening
from .game import BLACK, EMPTY, Board
from .mcts import MCTSConfig, search_many
from .model import GomokuNet
from .self_play import NeuralEvaluator

_ELO_SCALE = math.log(10.0) / 400.0


@dataclass(frozen=True, slots=True)
class TournamentGame:
    game_id: int
    matchup: str
    opening_pair: int
    seed: int
    black: str
    white: str
    opening: tuple[int, ...]
    moves: tuple[int, ...]
    outcome: int

    def score_for(self, label: str) -> float:
        if label not in (self.black, self.white):
            raise ValueError(f"model {label!r} did not play game {self.game_id}")
        if self.outcome == EMPTY:
            return 0.5
        winner = self.black if self.outcome == BLACK else self.white
        return 1.0 if label == winner else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "game_id": self.game_id,
            "matchup": self.matchup,
            "opening_pair": self.opening_pair,
            "seed": self.seed,
            "black": self.black,
            "white": self.white,
            "opening": list(self.opening),
            "moves": list(self.moves),
            "outcome": self.outcome,
            "result": "draw" if self.outcome == EMPTY else (
                "black_win" if self.outcome == BLACK else "white_win"
            ),
        }


@dataclass(slots=True)
class _GameState:
    game_id: int
    matchup: str
    opening_pair: int
    seed: int
    black: str
    white: str
    opening: tuple[int, ...]
    board: Board
    moves: list[int]
    rng: np.random.Generator

    @property
    def to_play_model(self) -> str:
        return self.black if self.board.to_play == BLACK else self.white


def _sha256(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(json.dumps(dict(payload), sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _board_after_opening(opening: Sequence[int]) -> Board:
    board = Board.empty()
    for action in opening:
        if not board.is_legal(action):
            raise ValueError(f"opening contains illegal action {action}")
        board = board.play(action)
    if board.is_terminal:
        raise ValueError("tournament opening must not be terminal")
    return board


def build_round_robin_schedule(
    labels: Sequence[str],
    *,
    games_per_pair: int,
    opening_plies: int,
    seed: int,
) -> tuple[_GameState, ...]:
    """Create random-first, exactly color-balanced paired-opening games."""

    names = tuple(labels)
    if len(names) < 2 or len(set(names)) != len(names):
        raise ValueError("at least two unique model labels are required")
    if games_per_pair <= 0 or games_per_pair % 2:
        raise ValueError("games_per_pair must be a positive even number")
    if opening_plies < 0:
        raise ValueError("opening_plies must be non-negative")

    scheduled: list[_GameState] = []
    game_id = 0
    for matchup_index, (left, right) in enumerate(itertools.combinations(names, 2)):
        matchup = f"{left}__vs__{right}"
        for opening_pair in range(games_per_pair // 2):
            sequence = np.random.SeedSequence([seed, matchup_index, opening_pair])
            game_seed = int(sequence.generate_state(1, dtype=np.uint64)[0])
            opening = generate_fixed_opening(game_seed, opening_plies)
            color_rng = np.random.default_rng(
                np.random.SeedSequence([seed, matchup_index, opening_pair, 1])
            )
            left_black_first = bool(color_rng.integers(0, 2))
            first_black, first_white = (
                (left, right) if left_black_first else (right, left)
            )
            for black, white in (
                (first_black, first_white),
                (first_white, first_black),
            ):
                scheduled.append(
                    _GameState(
                        game_id=game_id,
                        matchup=matchup,
                        opening_pair=opening_pair,
                        seed=game_seed,
                        black=black,
                        white=white,
                        opening=opening,
                        board=_board_after_opening(opening),
                        moves=list(opening),
                        rng=np.random.default_rng(game_seed),
                    )
                )
                game_id += 1
    return tuple(scheduled)


def play_round_robin(
    models: Mapping[str, torch.nn.Module],
    *,
    games_per_pair: int,
    simulations: int,
    opening_plies: int,
    seed: int,
    device: str | torch.device,
    max_batch_size: int,
    c_puct: float = 1.5,
    on_game: Callable[[TournamentGame], None] | None = None,
) -> tuple[tuple[TournamentGame, ...], dict[str, Any]]:
    if simulations <= 0 or max_batch_size <= 0:
        raise ValueError("simulations and max_batch_size must be positive")
    labels = tuple(models)
    active = list(
        build_round_robin_schedule(
            labels,
            games_per_pair=games_per_pair,
            opening_plies=opening_plies,
            seed=seed,
        )
    )
    evaluators = {
        label: NeuralEvaluator(model, device=device, mixed_precision=True)
        for label, model in models.items()
    }
    completed: list[TournamentGame] = []
    started = time.perf_counter()
    while active:
        groups: dict[str, list[_GameState]] = defaultdict(list)
        for state in active:
            groups[state.to_play_model].append(state)

        searched: dict[int, Any] = {}
        for label in labels:
            states = groups.get(label, [])
            if not states:
                continue
            configs = tuple(
                MCTSConfig(
                    c_puct=c_puct,
                    add_root_noise=False,
                    seed=int(state.rng.integers(0, 2**63)),
                )
                for state in states
            )
            results = search_many(
                tuple(state.board for state in states),
                evaluators[label],
                simulations,
                configs,
                max_batch_size=max_batch_size,
            )
            for state, result in zip(states, results, strict=True):
                searched[state.game_id] = result

        survivors: list[_GameState] = []
        for state in active:
            result = searched[state.game_id]
            action = result.best_move
            if action is None or not state.board.is_legal(action):
                raise RuntimeError("tournament search did not produce a legal move")
            state.moves.append(action)
            state.board = state.board.play(action)
            if state.board.is_terminal:
                outcome = state.board.outcome
                if outcome is None:  # pragma: no cover - terminal guard
                    raise RuntimeError("terminal tournament game has no outcome")
                game = TournamentGame(
                    game_id=state.game_id,
                    matchup=state.matchup,
                    opening_pair=state.opening_pair,
                    seed=state.seed,
                    black=state.black,
                    white=state.white,
                    opening=state.opening,
                    moves=tuple(state.moves),
                    outcome=outcome,
                )
                completed.append(game)
                if on_game is not None:
                    on_game(game)
            else:
                survivors.append(state)
        active = survivors

    elapsed = time.perf_counter() - started
    telemetry = {
        "seconds": elapsed,
        "games": len(completed),
        "games_per_hour": len(completed) * 3600.0 / elapsed if elapsed else 0.0,
        "average_plies": float(np.mean([len(game.moves) for game in completed])),
        "models": {label: evaluator.telemetry() for label, evaluator in evaluators.items()},
    }
    return tuple(sorted(completed, key=lambda game: game.game_id)), telemetry


def _fit_elo_point(
    labels: Sequence[str],
    games: Sequence[TournamentGame],
    *,
    anchor: str,
    prior_sigma: float = 800.0,
) -> dict[str, float]:
    names = tuple(labels)
    if anchor not in names:
        raise ValueError("anchor must be one of the model labels")
    free = tuple(label for label in names if label != anchor)
    free_index = {label: index for index, label in enumerate(free)}
    ratings = np.full(len(free), 1000.0, dtype=np.float64)
    prior_precision = 1.0 / (prior_sigma * prior_sigma)

    for _ in range(100):
        gradient = prior_precision * (ratings - 1000.0)
        hessian = np.eye(len(free), dtype=np.float64) * prior_precision
        for game in games:
            black_rating = 1000.0 if game.black == anchor else ratings[free_index[game.black]]
            white_rating = 1000.0 if game.white == anchor else ratings[free_index[game.white]]
            logit = _ELO_SCALE * (black_rating - white_rating)
            expected = 1.0 / (1.0 + math.exp(-max(-50.0, min(50.0, logit))))
            observed = game.score_for(game.black)
            error = expected - observed
            weight = _ELO_SCALE * _ELO_SCALE * expected * (1.0 - expected)
            black_index = free_index.get(game.black)
            white_index = free_index.get(game.white)
            if black_index is not None:
                gradient[black_index] += _ELO_SCALE * error
                hessian[black_index, black_index] += weight
            if white_index is not None:
                gradient[white_index] -= _ELO_SCALE * error
                hessian[white_index, white_index] += weight
            if black_index is not None and white_index is not None:
                hessian[black_index, white_index] -= weight
                hessian[white_index, black_index] -= weight
        step = np.linalg.solve(hessian, gradient)
        ratings -= step
        if float(np.max(np.abs(step))) < 1e-7:
            break
    return {anchor: 1000.0, **dict(zip(free, ratings.tolist(), strict=True))}


def fit_elo(
    labels: Sequence[str],
    games: Sequence[TournamentGame],
    *,
    anchor: str,
    bootstrap_samples: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    if not games:
        raise ValueError("cannot fit Elo without games")
    point = _fit_elo_point(labels, games, anchor=anchor)
    by_matchup: dict[str, dict[int, list[TournamentGame]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for game in games:
        by_matchup[game.matchup][game.opening_pair].append(game)
    rng = np.random.default_rng(seed)
    samples = {label: [] for label in labels}
    for _ in range(bootstrap_samples):
        resampled: list[TournamentGame] = []
        for units in by_matchup.values():
            keys = tuple(sorted(units))
            picks = rng.integers(0, len(keys), size=len(keys))
            for pick in picks:
                resampled.extend(units[keys[int(pick)]])
        fitted = _fit_elo_point(labels, resampled, anchor=anchor)
        for label in labels:
            samples[label].append(fitted[label])
    rows = []
    for label in labels:
        values = np.asarray(samples[label], dtype=np.float64)
        rows.append(
            {
                "model": label,
                "elo": point[label],
                "bootstrap_95_interval": [
                    float(np.quantile(values, 0.025)),
                    float(np.quantile(values, 0.975)),
                ],
            }
        )
    rows.sort(key=lambda row: (-float(row["elo"]), str(row["model"])))
    return {
        "anchor": anchor,
        "anchor_elo": 1000.0,
        "method": "regularized logistic MLE; draw=0.5; paired-opening cluster bootstrap",
        "bootstrap_samples": bootstrap_samples,
        "ratings": rows,
    }


def summarize_pairings(
    labels: Sequence[str], games: Sequence[TournamentGame]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for left, right in itertools.combinations(labels, 2):
        selected = [game for game in games if {game.black, game.white} == {left, right}]
        left_scores = [game.score_for(left) for game in selected]
        rows.append(
            {
                "left": left,
                "right": right,
                "games": len(selected),
                "left_wins": sum(score == 1.0 for score in left_scores),
                "draws": sum(score == 0.5 for score in left_scores),
                "left_losses": sum(score == 0.0 for score in left_scores),
                "left_score": float(np.mean(left_scores)),
                "left_black_games": sum(game.black == left for game in selected),
                "right_black_games": sum(game.black == right for game in selected),
            }
        )
    return rows


def run_tournament(
    checkpoint_paths: Mapping[str, str | os.PathLike[str]],
    output_dir: str | os.PathLike[str],
    *,
    games_per_pair: int,
    simulations: int,
    opening_plies: int,
    seed: int,
    device: str | torch.device,
    max_batch_size: int,
    bootstrap_samples: int,
    anchor: str,
) -> Path:
    output = Path(output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"tournament output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    labels = tuple(checkpoint_paths)
    device_value = torch.device(device)
    models = {
        label: GomokuNet.from_checkpoint(path, map_location=device_value).to(device_value)
        for label, path in checkpoint_paths.items()
    }
    manifest = {
        "schema_version": 1,
        "seed": seed,
        "games_per_pair": games_per_pair,
        "simulations_per_move": simulations,
        "opening_plies": opening_plies,
        "first_player_randomization": (
            "random coin flip for the first game of each opening pair; exact color swap second"
        ),
        "max_batch_size": max_batch_size,
        "device": str(device_value),
        "anchor": anchor,
        "models": {
            label: {
                "path": str(Path(path).expanduser().resolve()),
                "sha256": _sha256(path),
            }
            for label, path in checkpoint_paths.items()
        },
    }
    _atomic_json(manifest, output / "manifest.json")
    games_path = output / "games.jsonl"

    def append(game: TournamentGame) -> None:
        with games_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(game.as_dict(), sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    games, telemetry = play_round_robin(
        models,
        games_per_pair=games_per_pair,
        simulations=simulations,
        opening_plies=opening_plies,
        seed=seed,
        device=device_value,
        max_batch_size=max_batch_size,
        on_game=append,
    )
    elo = fit_elo(
        labels,
        games,
        anchor=anchor,
        bootstrap_samples=bootstrap_samples,
        seed=seed + 1,
    )
    summary = {
        **manifest,
        "games": len(games),
        "opening_pairs": len(games) // 2,
        "pairings": summarize_pairings(labels, games),
        "elo": elo,
        "telemetry": telemetry,
        "interpretation": (
            "Elo is relative to the anchored model and this fixed search budget; "
            "small-sample intervals can be wide and are not universal ratings."
        ),
    }
    summary_path = output / "summary.json"
    _atomic_json(summary, summary_path)
    return summary_path


def _parse_model(value: str) -> tuple[str, Path]:
    label, separator, path = value.partition("=")
    if not separator or not label or not path:
        raise argparse.ArgumentTypeError("model must use LABEL=PATH")
    if not label.replace("_", "").replace("-", "").isalnum():
        raise argparse.ArgumentTypeError("model label must be alphanumeric with _ or -")
    checkpoint = Path(path).expanduser().resolve()
    if not checkpoint.is_file():
        raise argparse.ArgumentTypeError(f"checkpoint does not exist: {checkpoint}")
    return label, checkpoint


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", required=True, type=_parse_model)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--games-per-pair", type=int, default=8)
    parser.add_argument("--simulations", type=int, default=128)
    parser.add_argument("--opening-plies", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-batch-size", type=int, default=32)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--anchor")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    model_items = args.model
    checkpoints = dict(model_items)
    if len(checkpoints) != len(model_items):
        raise SystemExit("model labels must be unique")
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    anchor = args.anchor or next(iter(checkpoints))
    summary = run_tournament(
        checkpoints,
        args.output_dir,
        games_per_pair=args.games_per_pair,
        simulations=args.simulations,
        opening_plies=args.opening_plies,
        seed=args.seed,
        device=device,
        max_batch_size=args.max_batch_size,
        bootstrap_samples=args.bootstrap_samples,
        anchor=anchor,
    )
    print(json.dumps({"event": "complete", "summary": str(summary)}), flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "TournamentGame",
    "build_round_robin_schedule",
    "fit_elo",
    "main",
    "play_round_robin",
    "run_tournament",
    "summarize_pairings",
]
