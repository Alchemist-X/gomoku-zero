"""Deterministic AlphaZero self-play with CPU actors or batched GPU lanes."""

from __future__ import annotations

import multiprocessing as mp
import time
from collections.abc import Iterable, Mapping, MutableMapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch

from .config import ModelConfig, TrainingConfig, seed_everything
from .game import BOARD_CELLS, Board
from .mcts import MCTSConfig, search, search_many
from .model import masked_softmax
from .replay import TrainingSample, all_symmetries, normalize_policy_target


class NeuralEvaluator:
    """Adapt a PyTorch policy/WDL network to the MCTS probability contract."""

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        device: str | torch.device = "cpu",
        mixed_precision: bool = False,
    ) -> None:
        self.model = model
        self.device = torch.device(device)
        self.mixed_precision = mixed_precision
        # train_steps() leaves the shared model in training mode.  Eval mode is
        # mandatory here: BatchNorm must not depend on lane composition.
        self.model.eval()
        self.forward_calls = 0
        self.positions_evaluated = 0
        self.max_observed_batch_size = 0
        self.inference_seconds = 0.0

    def __call__(self, board: Board) -> tuple[np.ndarray, np.ndarray]:
        return self.evaluate_batch((board,))[0]

    def evaluate_batch(
        self,
        boards: Sequence[Board],
    ) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
        """Evaluate all active positions in one ``[B, 3, 15, 15]`` forward.

        Terminal rows are completed exactly on the CPU and never sent to the
        network.  For active rows, occupied points are masked before the
        batched softmax, so each returned policy has exactly zero illegal mass.
        """

        board_items = tuple(boards)
        if not board_items:
            return ()

        results: list[tuple[np.ndarray, np.ndarray] | None] = [None] * len(board_items)
        active_indices: list[int] = []
        active_states: list[np.ndarray] = []
        active_masks: list[np.ndarray] = []
        for index, board in enumerate(board_items):
            legal = np.asarray(board.legal_mask, dtype=np.bool_).reshape(-1)
            if legal.size != BOARD_CELLS:
                raise ValueError(
                    f"board legal mask must contain {BOARD_CELLS} entries, got {legal.size}"
                )
            if np.any(legal):
                active_indices.append(index)
                active_states.append(board.encode())
                active_masks.append(legal)
            else:
                results[index] = (
                    np.zeros(BOARD_CELLS, dtype=np.float64),
                    board.wdl_target().astype(np.float64),
                )
        if active_indices:
            started = time.perf_counter()
            states = torch.as_tensor(
                np.stack(active_states),
                dtype=torch.float32,
                device=self.device,
            )
            legal_tensor = torch.as_tensor(
                np.stack(active_masks),
                dtype=torch.bool,
                device=self.device,
            )
            with (
                torch.inference_mode(),
                torch.autocast(
                    device_type=self.device.type,
                    enabled=self.mixed_precision and self.device.type == "cuda",
                ),
            ):
                policy_logits, wdl_logits = self.model(states)
            expected_policy_shape = (len(active_indices), BOARD_CELLS)
            expected_wdl_shape = (len(active_indices), 3)
            if tuple(policy_logits.shape) != expected_policy_shape:
                raise ValueError(
                    "model policy output must have shape "
                    f"{expected_policy_shape}, got {tuple(policy_logits.shape)}"
                )
            if tuple(wdl_logits.shape) != expected_wdl_shape:
                raise ValueError(
                    f"model WDL output must have shape {expected_wdl_shape}, "
                    f"got {tuple(wdl_logits.shape)}"
                )
            if not torch.isfinite(wdl_logits).all():
                raise ValueError("model returned NaN or infinite WDL logits")

            policies = masked_softmax(policy_logits.float(), legal_tensor, dim=-1)
            wdls = torch.softmax(wdl_logits.float(), dim=-1)
            if not torch.isfinite(wdls).all():
                raise ValueError("model returned invalid WDL probabilities")
            # One device-to-host transfer per output tensor, not one per lane.
            policies_cpu = policies.to(torch.float64).cpu().numpy()
            wdls_cpu = wdls.to(torch.float64).cpu().numpy()
            elapsed = time.perf_counter() - started
            batch_size = len(active_indices)
            self.forward_calls += 1
            self.positions_evaluated += batch_size
            self.max_observed_batch_size = max(self.max_observed_batch_size, batch_size)
            self.inference_seconds += elapsed
            for output_index, board_index in enumerate(active_indices):
                results[board_index] = (policies_cpu[output_index], wdls_cpu[output_index])

        if any(item is None for item in results):  # pragma: no cover - defensive
            raise RuntimeError("batch evaluator failed to produce every requested result")
        return tuple(item for item in results if item is not None)

    def telemetry(self) -> dict[str, int | float]:
        mean_batch = self.positions_evaluated / self.forward_calls if self.forward_calls else 0.0
        return {
            "inference_calls": self.forward_calls,
            "positions_evaluated": self.positions_evaluated,
            "mean_effective_batch_size": mean_batch,
            "max_effective_batch_size": self.max_observed_batch_size,
            "inference_seconds": self.inference_seconds,
            "inference_positions_per_second": (
                self.positions_evaluated / self.inference_seconds
                if self.inference_seconds > 0.0
                else 0.0
            ),
        }


@dataclass(frozen=True, slots=True)
class SelfPlayGame:
    seed: int
    samples: tuple[TrainingSample, ...]
    outcome: int
    moves: tuple[int, ...]


@dataclass(slots=True)
class _BatchedGameState:
    output_index: int
    seed: int
    rng: np.random.Generator
    board: Board
    positions: list[tuple[np.ndarray, np.ndarray, np.ndarray]]
    moves: list[int]


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


def resolve_self_play_backend(
    config: TrainingConfig,
    device: str | torch.device,
) -> str:
    """Return the explicit, checkpoint-stable self-play backend."""

    del device
    return config.self_play_backend


def _finish_batched_game(state: _BatchedGameState, config: TrainingConfig) -> SelfPlayGame:
    wdl = state.board.wdl_target()
    samples: list[TrainingSample] = []
    for encoded_state, policy, legal_mask in state.positions:
        sample = TrainingSample(encoded_state, policy, wdl, legal_mask)
        if config.symmetry_augmentation == "all":
            samples.extend(all_symmetries(sample))
        else:
            samples.append(sample)
    outcome = state.board.outcome
    if outcome is None:  # pragma: no cover - caller checks terminal state
        raise RuntimeError("self-play finished without a terminal outcome")
    return SelfPlayGame(
        seed=state.seed,
        samples=tuple(samples),
        outcome=outcome,
        moves=tuple(state.moves),
    )


def _play_games_batched(
    model: torch.nn.Module,
    config: TrainingConfig,
    seeds: Sequence[int],
    *,
    device: str | torch.device,
    stats: MutableMapping[str, Any] | None = None,
) -> list[SelfPlayGame]:
    """Keep independent games in lanes and coalesce one leaf per tree."""

    seed_items = tuple(int(seed) for seed in seeds)
    if not seed_items:
        return []
    evaluator = NeuralEvaluator(
        model,
        device=device,
        mixed_precision=config.mixed_precision,
    )
    lane_count = min(config.self_play_lanes, len(seed_items))
    completed: list[SelfPlayGame | None] = [None] * len(seed_items)
    next_index = 0
    active: list[_BatchedGameState] = []

    def add_lane(index: int) -> None:
        seed = seed_items[index]
        active.append(
            _BatchedGameState(
                output_index=index,
                seed=seed,
                rng=np.random.default_rng(seed),
                board=Board.empty(),
                positions=[],
                moves=[],
            )
        )

    while next_index < lane_count:
        add_lane(next_index)
        next_index += 1

    started = time.perf_counter()
    while active:
        search_configs: list[MCTSConfig] = []
        for state in active:
            search_seed = int(state.rng.integers(0, np.iinfo(np.int64).max))
            search_configs.append(
                MCTSConfig(
                    c_puct=config.c_puct,
                    dirichlet_alpha=config.dirichlet_alpha,
                    dirichlet_epsilon=config.dirichlet_fraction,
                    add_root_noise=True,
                    seed=search_seed,
                )
            )
        results = search_many(
            tuple(state.board for state in active),
            evaluator,
            config.mcts_simulations,
            tuple(search_configs),
            max_batch_size=config.inference_batch_size,
        )

        survivors: list[_BatchedGameState] = []
        for state, result in zip(active, results, strict=True):
            legal = state.board.legal_mask
            target = normalize_policy_target(result.visit_policy, legal)
            state.positions.append((state.board.encode(), target, legal))

            temperature = 1.0 if len(state.moves) < config.temperature_moves else 0.0
            move_policy = normalize_policy_target(
                result.policy_at_temperature(temperature),
                legal,
            )
            action = sample_legal_action(
                move_policy,
                legal,
                state.rng,
                deterministic=temperature == 0.0,
            )
            if not state.board.is_legal(action):
                raise RuntimeError(f"MCTS selected illegal action {action}")
            state.moves.append(action)
            state.board = state.board.play(action)
            if state.board.is_terminal:
                completed[state.output_index] = _finish_batched_game(state, config)
            else:
                survivors.append(state)

        active = survivors
        while len(active) < lane_count and next_index < len(seed_items):
            add_lane(next_index)
            next_index += 1

    elapsed = time.perf_counter() - started
    if stats is not None:
        stats.update(
            {
                "backend": "batched",
                "device": str(torch.device(device)),
                "games": len(seed_items),
                "lanes": lane_count,
                "configured_inference_batch_size": config.inference_batch_size,
                "self_play_seconds": elapsed,
                **evaluator.telemetry(),
            }
        )
        positions = int(stats["positions_evaluated"])
        stats["end_to_end_positions_per_second"] = positions / elapsed if elapsed > 0.0 else 0.0
    if any(game is None for game in completed):  # pragma: no cover - defensive
        raise RuntimeError("batched self-play did not complete every game")
    return [game for game in completed if game is not None]


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
    stats: MutableMapping[str, Any] | None = None,
) -> list[SelfPlayGame]:
    """Generate games through deterministic CPU actors or batched lanes.

    Actor scheduling cannot change samples because every game receives its own
    seed.  CPU actors reconstruct a frozen model from a serialized state dict;
    the training model and optimizer remain exclusively in the parent process.
    Batched mode keeps independent PUCT trees in the parent and executes one
    shared network forward for up to ``inference_batch_size`` leaves.
    """

    game_seeds = tuple(int(seed) for seed in seeds)
    worker_count = training_config.self_play_actors if actors is None else actors
    if worker_count <= 0:
        raise ValueError("actors must be > 0")
    if not game_seeds:
        return []
    backend = resolve_self_play_backend(training_config, device)
    if backend == "batched":
        if actors is not None:
            raise ValueError(
                "actors override applies only to the process backend; "
                "configure training.self_play_lanes for batched self-play"
            )
        return _play_games_batched(
            model,
            training_config,
            game_seeds,
            device=device,
            stats=stats,
        )

    started = time.perf_counter()
    if worker_count == 1:
        games = [
            _play_one_game(model, training_config, seed=seed, device=device) for seed in game_seeds
        ]
    else:
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
            games = list(executor.map(_actor_play, game_seeds, chunksize=1))
    if stats is not None:
        stats.update(
            {
                "backend": "process",
                "device": "cpu" if worker_count > 1 else str(torch.device(device)),
                "games": len(game_seeds),
                "actors": worker_count,
                "configured_inference_batch_size": 1,
                "self_play_seconds": time.perf_counter() - started,
                "batch_telemetry_available": False,
            }
        )
    return games


__all__ = [
    "NeuralEvaluator",
    "SelfPlayGame",
    "derive_game_seeds",
    "generate_self_play_games",
    "play_self_play_game",
    "resolve_self_play_backend",
    "sample_legal_action",
]
