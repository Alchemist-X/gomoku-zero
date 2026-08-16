"""Fixed-policy evaluation, statistical reporting, sharding, and resume."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
import torch

from .config import EvaluationConfig, RunConfig, load_config, seed_everything
from .game import BLACK, EMPTY, WHITE, Board
from .mcts import MCTSConfig, search
from .model import GomokuNet
from .self_play import NeuralEvaluator


def wilson_interval(successes: int, trials: int, confidence: float = 0.95) -> tuple[float, float]:
    """Two-sided Wilson score interval for a binomial proportion."""

    if trials <= 0:
        raise ValueError("trials must be > 0")
    if not 0 <= successes <= trials:
        raise ValueError("successes must fall between zero and trials")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    proportion = successes / trials
    z2 = z * z
    denominator = 1.0 + z2 / trials
    center = (proportion + z2 / (2.0 * trials)) / denominator
    margin = (
        z
        * math.sqrt(proportion * (1.0 - proportion) / trials + z2 / (4.0 * trials * trials))
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


binomial_confidence_interval = wilson_interval


@dataclass(frozen=True, slots=True)
class EvaluationGame:
    game_id: int
    seed: int
    outcome: int
    moves: tuple[int, ...]
    opening: tuple[int, ...]
    candidate_color: int | None = None

    @property
    def candidate_result(self) -> str | None:
        if self.candidate_color is None:
            return None
        if self.outcome == EMPTY:
            return "draw"
        return "win" if self.outcome == self.candidate_color else "loss"

    def as_dict(self) -> dict[str, Any]:
        return {
            "game_id": self.game_id,
            "seed": self.seed,
            "outcome": self.outcome,
            "moves": list(self.moves),
            "opening": list(self.opening),
            "candidate_color": self.candidate_color,
            "candidate_result": self.candidate_result,
        }


def _seed_for_game(seed: int, game_id: int, *, paired: bool) -> int:
    opening_id = game_id // 2 if paired else game_id
    sequence = np.random.SeedSequence([seed, opening_id])
    return int(sequence.generate_state(1, dtype=np.uint64)[0])


def generate_fixed_opening(seed: int, plies: int) -> tuple[int, ...]:
    """Generate a reproducible legal opening without consulting either model."""

    if plies < 0:
        raise ValueError("opening plies must be >= 0")
    rng = np.random.default_rng(seed)
    board = Board.empty()
    opening: list[int] = []
    for _ in range(plies):
        if board.is_terminal:
            break
        legal = board.legal_moves
        action = int(legal[int(rng.integers(0, len(legal)))])
        opening.append(action)
        board = board.play(action)
    return tuple(opening)


def play_fixed_policy_game(
    black_model: torch.nn.Module,
    white_model: torch.nn.Module,
    *,
    simulations: int,
    seed: int,
    game_id: int = 0,
    opening: Sequence[int] = (),
    c_puct: float = 1.5,
    device: str | torch.device = "cpu",
    candidate_color: int | None = None,
) -> EvaluationGame:
    """Play one noise-free, temperature-zero game from a fixed opening."""

    if simulations <= 0:
        raise ValueError("simulations must be > 0")
    seed_everything(seed)
    board = Board.empty()
    moves: list[int] = []
    for action in opening:
        if not board.is_legal(int(action)):
            raise ValueError(f"opening contains illegal action {action}")
        moves.append(int(action))
        board = board.play(int(action))
    black_evaluator = NeuralEvaluator(black_model, device=device)
    white_evaluator = NeuralEvaluator(white_model, device=device)
    rng = np.random.default_rng(seed)
    while not board.is_terminal:
        evaluator = black_evaluator if board.to_play == BLACK else white_evaluator
        result = search(
            board,
            evaluator,
            simulations,
            MCTSConfig(c_puct=c_puct, add_root_noise=False, seed=int(rng.integers(0, 2**63))),
        )
        action = result.best_move
        if action is None or not board.is_legal(action):
            raise RuntimeError("fixed-policy search did not produce a legal move")
        moves.append(action)
        board = board.play(action)
    outcome = board.outcome
    if outcome is None:  # pragma: no cover - guarded by loop condition
        raise RuntimeError("evaluation game ended without an outcome")
    return EvaluationGame(
        game_id=game_id,
        seed=seed,
        outcome=outcome,
        moves=tuple(moves),
        opening=tuple(int(action) for action in opening),
        candidate_color=candidate_color,
    )


def summarize_games(
    games: Iterable[EvaluationGame], confidence: float = 0.95
) -> dict[str, Any]:
    items = list(games)
    if not items:
        raise ValueError("cannot summarize zero games")
    counts = {
        "black": sum(game.outcome == BLACK for game in items),
        "draw": sum(game.outcome == EMPTY for game in items),
        "white": sum(game.outcome == WHITE for game in items),
    }
    total = len(items)
    result: dict[str, Any] = {"games": total, "confidence_level": confidence}
    for label, count in counts.items():
        low, high = wilson_interval(count, total, confidence)
        result[label] = {
            "count": count,
            "rate": count / total,
            "wilson_interval": [low, high],
        }

    candidate_games = [game for game in items if game.candidate_color is not None]
    if candidate_games:
        wins = sum(game.candidate_result == "win" for game in candidate_games)
        draws = sum(game.candidate_result == "draw" for game in candidate_games)
        losses = len(candidate_games) - wins - draws
        win_low, win_high = wilson_interval(wins, len(candidate_games), confidence)
        result["candidate"] = {
            "wins": wins,
            "draws": draws,
            "losses": losses,
            "win_rate": wins / len(candidate_games),
            "win_rate_wilson_interval": [win_low, win_high],
            "candidate_score": (wins + 0.5 * draws) / len(candidate_games),
        }
    return result


def evaluate_head_to_head(
    candidate: torch.nn.Module,
    champion: torch.nn.Module,
    *,
    games: int,
    simulations: int,
    seed: int,
    c_puct: float = 1.5,
    opening_plies: int = 4,
    paired_openings: bool = True,
    confidence: float = 0.95,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Evaluate a candidate with balanced colors and fixed paired openings."""

    if games <= 0:
        raise ValueError("games must be > 0")
    results: list[EvaluationGame] = []
    for game_id in range(games):
        game_seed = _seed_for_game(seed, game_id, paired=paired_openings)
        opening = generate_fixed_opening(game_seed, opening_plies)
        candidate_color = BLACK if game_id % 2 == 0 else WHITE
        black = candidate if candidate_color == BLACK else champion
        white = candidate if candidate_color == WHITE else champion
        results.append(
            play_fixed_policy_game(
                black,
                white,
                simulations=simulations,
                seed=game_seed,
                game_id=game_id,
                opening=opening,
                c_puct=c_puct,
                device=device,
                candidate_color=candidate_color,
            )
        )
    summary = summarize_games(results, confidence)
    summary.update(summary["candidate"])
    return summary


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


def _load_existing(path: Path) -> dict[int, EvaluationGame]:
    existing: dict[int, EvaluationGame] = {}
    if not path.exists():
        return existing
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
            game = EvaluationGame(
                game_id=int(raw["game_id"]),
                seed=int(raw["seed"]),
                outcome=int(raw["outcome"]),
                moves=tuple(int(value) for value in raw["moves"]),
                opening=tuple(int(value) for value in raw["opening"]),
                candidate_color=(
                    None if raw.get("candidate_color") is None else int(raw["candidate_color"])
                ),
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid evaluation JSONL at line {line_number}") from exc
        if game.game_id in existing:
            raise ValueError(f"duplicate evaluation game_id {game.game_id}")
        existing[game.game_id] = game
    return existing


def _append_game(path: Path, game: EvaluationGame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(game.as_dict(), sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def evaluation_game_ids(
    games: int,
    *,
    start_game: int = 0,
    shard_index: int = 0,
    num_shards: int = 1,
    paired_openings: bool = False,
) -> tuple[int, ...]:
    if games <= 0 or start_game < 0:
        raise ValueError("games must be positive and start_game non-negative")
    if num_shards <= 0 or not 0 <= shard_index < num_shards:
        raise ValueError("shard_index must fall in [0, num_shards)")
    if paired_openings and (start_game % 2 or games % 2):
        raise ValueError("paired-opening evaluation requires an even start and game count")
    return tuple(
        game_id
        for game_id in range(start_game, start_game + games)
        if (game_id // 2 if paired_openings else game_id) % num_shards == shard_index
    )


def _file_sha256(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def run_evaluation(
    config: RunConfig,
    checkpoint: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    *,
    opponent_checkpoint: str | os.PathLike[str] | None = None,
    games: int | None = None,
    simulations: int | None = None,
    start_game: int = 0,
    shard_index: int = 0,
    num_shards: int = 1,
    resume: bool = False,
    device: str | torch.device = "cpu",
) -> Path:
    evaluation = config.evaluation
    game_count = evaluation.games if games is None else games
    simulation_count = evaluation.mcts_simulations if simulations is None else simulations
    # Pairing is useful only when candidate/opponent colors are swapped.  A
    # single fixed policy instead receives one independently seeded opening per
    # game, so 6000 games do not collapse into 3000 exact duplicates.
    paired = evaluation.paired_openings and opponent_checkpoint is not None
    ids = evaluation_game_ids(
        game_count,
        start_game=start_game,
        shard_index=shard_index,
        num_shards=num_shards,
        paired_openings=paired,
    )
    if not ids:
        raise ValueError("this shard contains no selected games")
    device_value = torch.device(device)
    model = GomokuNet.from_checkpoint(checkpoint, map_location=device_value).to(device_value)
    opponent = (
        GomokuNet.from_checkpoint(opponent_checkpoint, map_location=device_value).to(device_value)
        if opponent_checkpoint is not None
        else model
    )
    output = Path(output_dir).expanduser().resolve()
    results_path = output / f"games-shard-{shard_index:04d}-of-{num_shards:04d}.jsonl"
    manifest_path = output / f"manifest-shard-{shard_index:04d}-of-{num_shards:04d}.json"
    config_json = json.dumps(config.to_dict(), sort_keys=True, separators=(",", ":"))
    manifest = {
        "checkpoint_sha256": _file_sha256(checkpoint),
        "opponent_checkpoint_sha256": (
            None if opponent_checkpoint is None else _file_sha256(opponent_checkpoint)
        ),
        "config_sha256": hashlib.sha256(config_json.encode()).hexdigest(),
        "mcts_simulations": simulation_count,
        "game_count": game_count,
        "start_game": start_game,
        "shard_index": shard_index,
        "num_shards": num_shards,
        "paired_openings": paired,
    }
    if resume:
        if not manifest_path.exists():
            raise FileNotFoundError("resume requested but evaluation manifest is missing")
        saved_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if saved_manifest != manifest:
            raise ValueError("evaluation manifest mismatch; refusing to mix policies/configs")
    else:
        if results_path.exists() or manifest_path.exists():
            raise FileExistsError(
                "evaluation shard already exists; pass --resume or choose a new output dir"
            )
        _atomic_json(manifest, manifest_path)
    existing = _load_existing(results_path) if resume else {}

    selected = set(ids)
    for game_id in ids:
        if game_id in existing:
            continue
        game_seed = _seed_for_game(config.seed, game_id, paired=paired)
        opening = generate_fixed_opening(game_seed, evaluation.opening_plies)
        # With an explicit opponent, checkpoint is the candidate and colors
        # alternate.  Without one this is a fixed-policy black/white estimate.
        candidate_color = None
        black_model, white_model = model, opponent
        if opponent_checkpoint is not None:
            candidate_color = BLACK if game_id % 2 == 0 else WHITE
            if candidate_color == WHITE:
                black_model, white_model = opponent, model
        result = play_fixed_policy_game(
            black_model,
            white_model,
            simulations=simulation_count,
            seed=game_seed,
            game_id=game_id,
            opening=opening,
            c_puct=config.training.c_puct,
            device=device_value,
            candidate_color=candidate_color,
        )
        _append_game(results_path, result)
        existing[game_id] = result

    summary = summarize_games(
        [existing[game_id] for game_id in sorted(selected)],
        evaluation.confidence_level,
    )
    summary.update(
        {
            "checkpoint": str(Path(checkpoint).expanduser().resolve()),
            "opponent_checkpoint": (
                None
                if opponent_checkpoint is None
                else str(Path(opponent_checkpoint).expanduser().resolve())
            ),
            "mcts_simulations": simulation_count,
            "opening_plies": evaluation.opening_plies,
            "paired_openings": paired,
            "position_scope": (
                "empty_board" if evaluation.opening_plies == 0 else "fixed_opening_suite"
            ),
            "is_empty_board_black_first_estimate": evaluation.opening_plies == 0,
            "independent_opening_units": (
                len({game_id // 2 for game_id in ids})
                if paired
                else len(ids)
            ),
            "confidence_interval_note": (
                "game-level Wilson intervals; paired-opening games are correlated"
                if paired
                else "game-level Wilson intervals"
            ),
            "shard_index": shard_index,
            "num_shards": num_shards,
            "first_game_id": min(ids),
            "last_game_id": max(ids),
        }
    )
    summary_path = output / f"summary-shard-{shard_index:04d}-of-{num_shards:04d}.json"
    _atomic_json(summary, summary_path)
    return summary_path


def estimate_evaluation_work(config: EvaluationConfig, games: int, simulations: int) -> int:
    return games * 225 * simulations


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selected = torch.device(value)
    if selected.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    return selected


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--opponent-checkpoint")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--games", type=int, help="number of global game IDs to select")
    parser.add_argument("--simulations", type=int, help="per-move MCTS budget")
    parser.add_argument("--start-game", type=int, default=0)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--resume", action="store_true", help="continue the shard JSONL")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--allow-large-run", action="store_true")
    parser.add_argument(
        "--allow-slow-evaluation",
        action="store_true",
        help="acknowledge batch=1 leaf evaluation for 1000+ games",
    )
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    games = config.evaluation.games if args.games is None else args.games
    simulations = (
        config.evaluation.mcts_simulations if args.simulations is None else args.simulations
    )
    if games <= 0 or simulations <= 0:
        raise SystemExit("games and simulations must be > 0")
    selected = evaluation_game_ids(
        games,
        start_game=args.start_game,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
        paired_openings=(
            config.evaluation.paired_openings and args.opponent_checkpoint is not None
        ),
    )
    estimate = estimate_evaluation_work(config.evaluation, len(selected), simulations)
    print(
        json.dumps(
            {
                "event": "preflight",
                "selected_games": len(selected),
                "mcts_simulations_per_move": simulations,
                "worst_case_leaf_evaluations": estimate,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if args.preflight_only:
        return 0
    if estimate > config.evaluation.max_estimated_simulations and not args.allow_large_run:
        raise SystemExit(
            "evaluation exceeds max_estimated_simulations; shard or pass --allow-large-run"
        )
    if games >= 1_000 and not args.allow_slow_evaluation:
        raise SystemExit(
            "large evaluation refused because leaves are evaluated at batch=1; "
            "use shards and pass --allow-slow-evaluation to acknowledge it"
        )
    summary_path = run_evaluation(
        config,
        args.checkpoint,
        args.output_dir,
        opponent_checkpoint=args.opponent_checkpoint,
        games=games,
        simulations=simulations,
        start_game=args.start_game,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
        resume=args.resume,
        device=_device(args.device),
    )
    print(json.dumps({"event": "complete", "summary": str(summary_path)}), flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "EvaluationGame",
    "binomial_confidence_interval",
    "estimate_evaluation_work",
    "evaluate_head_to_head",
    "evaluation_game_ids",
    "generate_fixed_opening",
    "main",
    "play_fixed_policy_game",
    "run_evaluation",
    "summarize_games",
    "wilson_interval",
]
