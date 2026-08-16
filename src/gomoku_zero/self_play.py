"""Deterministic AlphaZero self-play, including parallel CPU actors."""

from __future__ import annotations

import multiprocessing as mp
from collections.abc import Iterable, Mapping
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch

from .config import ModelConfig, TrainingConfig, seed_everything
from .game import BOARD_CELLS, Board
from .mcts import MCTSConfig, search
from .replay import TrainingSample, all_symmetries, normalize_policy_target


class NeuralEvaluator:
    """Adapt a PyTorch policy/WDL network to the MCTS probability contract."""

    def __init__(self, model: torch.nn.Module, *, device: str | torch.device = "cpu") -> None:
        self.model = model
        self.device = torch.device(device)

    def __call__(self, board: Board) -> tuple[np.ndarray, np.ndarray]:
        legal = board.legal_mask
        if not np.any(legal):
            return np.zeros(BOARD_CELLS, dtype=np.float64), board.wdl_target().astype(np.float64)
        self.model.eval()
        state = torch.as_tensor(
            board.encode(), dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.inference_mode():
            policy_logits, wdl_logits = self.model(state)
            policy_logits = policy_logits.reshape(-1)
            wdl_logits = wdl_logits.reshape(-1)
            if policy_logits.numel() != BOARD_CELLS:
                raise ValueError(
                    f"model policy output must have {BOARD_CELLS} logits, "
                    f"got {policy_logits.numel()}"
                )
            if wdl_logits.numel() != 3:
                raise ValueError(f"model WDL output must have 3 logits, got {wdl_logits.numel()}")
            if not torch.isfinite(policy_logits).all() or not torch.isfinite(wdl_logits).all():
                raise ValueError("model returned NaN or infinite logits")

            # Gather legal logits before normalization.  Occupied logits never
            # participate in a reduction and receive exactly zero probability.
            legal_tensor = torch.as_tensor(legal, dtype=torch.bool, device=self.device)
            legal_probabilities = torch.softmax(policy_logits.float()[legal_tensor], dim=0)
            policy = torch.zeros(BOARD_CELLS, dtype=torch.float64, device=self.device)
            policy[legal_tensor] = legal_probabilities.to(torch.float64)
            wdl = torch.softmax(wdl_logits.float(), dim=0).to(torch.float64)
        return policy.cpu().numpy(), wdl.cpu().numpy()


@dataclass(frozen=True, slots=True)
class SelfPlayGame:
    seed: int
    samples: tuple[TrainingSample, ...]
    outcome: int
    moves: tuple[int, ...]


def sample_legal_action(
    policy: np.ndarray,
    legal_mask: np.ndarray,
    rng: np.random.Generator,
    *,
    deterministic: bool = False,
) -> int:
    """Sample after an explicit float64 legal-only renormalization."""

    legal = np.flatnonzero(np.asarray(legal_mask, dtype=np.bool_).reshape(-1))
    if legal.size == 0:
        raise ValueError("cannot sample an action without legal moves")
    values = np.asarray(policy, dtype=np.float64).reshape(-1)[legal]
    if values.shape != (legal.size,) or not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("sampling policy contains invalid legal probabilities")
    total = float(values.sum(dtype=np.float64))
    if total <= 0.0:
        raise ValueError("sampling policy has no legal mass")
    values = values / total
    # numpy.random.Generator.choice uses a tighter sum tolerance than a float32
    # policy guarantees.  Make the float64 categorical sum exact by residual.
    values[-1] = 1.0 - float(values[:-1].sum(dtype=np.float64))
    if values[-1] < 0.0 and values[-1] > -1e-15:
        values[-1] = 0.0
        values /= values.sum(dtype=np.float64)
    if deterministic:
        return int(legal[int(np.argmax(values))])
    return int(rng.choice(legal, p=values))


def _play_one_game(
    model: torch.nn.Module,
    config: TrainingConfig,
    *,
    seed: int,
    device: str | torch.device = "cpu",
) -> SelfPlayGame:
    rng = seed_everything(seed)
    evaluator = NeuralEvaluator(model, device=device)
    board = Board.empty()
    positions: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    moves: list[int] = []

    while not board.is_terminal:
        # One independent, reproducible noisy root per move.  Deriving the MCTS
        # seed from this game's local generator decouples games from scheduling.
        search_seed = int(rng.integers(0, np.iinfo(np.int64).max))
        result = search(
            board,
            evaluator,
            config.mcts_simulations,
            MCTSConfig(
                c_puct=config.c_puct,
                dirichlet_alpha=config.dirichlet_alpha,
                dirichlet_epsilon=config.dirichlet_fraction,
                add_root_noise=True,
                seed=search_seed,
            ),
        )
        legal = board.legal_mask
        # The training target is the untempered legal visit distribution.  Move
        # sampling temperature affects exploration, not the supervised target.
        target = normalize_policy_target(result.visit_policy, legal)
        positions.append((board.encode(), target, legal))

        temperature = 1.0 if len(moves) < config.temperature_moves else 0.0
        move_policy = result.policy_at_temperature(temperature)
        move_policy = normalize_policy_target(move_policy, legal)
        action = sample_legal_action(
            move_policy,
            legal,
            rng,
            deterministic=temperature == 0.0,
        )
        if not board.is_legal(action):
            raise RuntimeError(f"MCTS selected illegal action {action}")
        moves.append(action)
        board = board.play(action)

    wdl = board.wdl_target()
    samples: list[TrainingSample] = []
    for encoded_state, policy, legal_mask in positions:
        sample = TrainingSample(encoded_state, policy, wdl, legal_mask)
        if config.symmetry_augmentation == "all":
            samples.extend(all_symmetries(sample))
        else:
            # Random augmentation happens at replay sampling time, avoiding an
            # eight-fold production buffer/checkpoint expansion.
            samples.append(sample)
    outcome = board.outcome
    if outcome is None:  # pragma: no cover - guarded by terminal loop condition
        raise RuntimeError("self-play finished without a terminal outcome")
    return SelfPlayGame(seed=seed, samples=tuple(samples), outcome=outcome, moves=tuple(moves))


def play_self_play_game(
    model: torch.nn.Module,
    config: TrainingConfig,
    *,
    seed: int,
    device: str | torch.device = "cpu",
) -> SelfPlayGame:
    """Public single-game entry point used by smoke tests and local runs."""

    return _play_one_game(model, config, seed=seed, device=device)


def derive_game_seeds(run_seed: int, iteration: int, games: int) -> tuple[int, ...]:
    """Derive schedule-independent seeds for an iteration's games."""

    if run_seed < 0 or iteration < 0 or games < 0:
        raise ValueError("run_seed, iteration, and games must be non-negative")
    sequence = np.random.SeedSequence([run_seed, iteration])
    children = sequence.spawn(games)
    return tuple(int(child.generate_state(1, dtype=np.uint64)[0]) for child in children)


_ACTOR_MODEL: torch.nn.Module | None = None
_ACTOR_TRAINING_CONFIG: TrainingConfig | None = None


def _construct_model(config: Mapping[str, Any]) -> torch.nn.Module:
    from .model import GomokuNet

    # Keep the constructor adapter narrow but tolerant of either keyword names
    # used by early versions of this project.
    try:
        return GomokuNet(
            channels=int(config["channels"]),
            residual_blocks=int(config["residual_blocks"]),
        )
    except TypeError:
        return GomokuNet(
            trunk_channels=int(config["channels"]),
            num_residual_blocks=int(config["residual_blocks"]),
        )


def _actor_initializer(
    state_dict: dict[str, torch.Tensor],
    model_config: dict[str, Any],
    training_config: dict[str, Any],
) -> None:
    global _ACTOR_MODEL, _ACTOR_TRAINING_CONFIG
    torch.set_num_threads(1)
    _ACTOR_MODEL = _construct_model(model_config)
    _ACTOR_MODEL.load_state_dict(state_dict, strict=True)
    _ACTOR_MODEL.eval()
    if isinstance(training_config.get("learning_rate_milestones"), list):
        training_config["learning_rate_milestones"] = tuple(
            training_config["learning_rate_milestones"]
        )
    _ACTOR_TRAINING_CONFIG = TrainingConfig(**training_config)


def _actor_play(seed: int) -> SelfPlayGame:
    if _ACTOR_MODEL is None or _ACTOR_TRAINING_CONFIG is None:
        raise RuntimeError("self-play actor was not initialized")
    return _play_one_game(_ACTOR_MODEL, _ACTOR_TRAINING_CONFIG, seed=seed, device="cpu")


def generate_self_play_games(
    model: torch.nn.Module,
    model_config: ModelConfig,
    training_config: TrainingConfig,
    seeds: Iterable[int],
    *,
    actors: int | None = None,
    device: str | torch.device = "cpu",
) -> list[SelfPlayGame]:
    """Generate games locally or through deterministic process actors.

    Actor scheduling cannot change samples because every game receives its own
    seed.  CPU actors reconstruct a frozen model from a serialized state dict;
    the training model and optimizer remain exclusively in the parent process.
    """

    game_seeds = tuple(int(seed) for seed in seeds)
    worker_count = training_config.self_play_actors if actors is None else actors
    if worker_count <= 0:
        raise ValueError("actors must be > 0")
    if not game_seeds:
        return []
    if worker_count == 1:
        return [
            _play_one_game(model, training_config, seed=seed, device=device)
            for seed in game_seeds
        ]

    cpu_state = {
        name: tensor.detach().to(device="cpu", copy=True)
        for name, tensor in model.state_dict().items()
    }
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=min(worker_count, len(game_seeds)),
        mp_context=context,
        initializer=_actor_initializer,
        initargs=(cpu_state, asdict(model_config), asdict(training_config)),
    ) as executor:
        return list(executor.map(_actor_play, game_seeds, chunksize=1))


__all__ = [
    "NeuralEvaluator",
    "SelfPlayGame",
    "derive_game_seeds",
    "generate_self_play_games",
    "play_self_play_game",
    "sample_legal_action",
]
