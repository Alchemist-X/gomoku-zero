"""Public FastAPI service for progressive Gomoku analysis.

The API deliberately separates two kinds of evidence:

* a checkpoint-backed network, when ``GOMOKU_CHECKPOINT`` (or a conventional
  ``checkpoints/best.pt`` file) is available; and
* a deterministic, legality-safe heuristic when no checkpoint exists.

The bootstrap heuristic keeps a fresh deployment useful, but its WDL-shaped
output is always labelled ``bootstrap-untrained`` and must not be presented as
training or evaluation statistics.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
import tempfile
import threading
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol
from urllib.parse import urlparse
from uuid import uuid4

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator

from . import __version__
from .game import BLACK, BOARD_CELLS, BOARD_SIZE, EMPTY, WHITE, Board
from .mcts import MCTSConfig, SearchResult, SearchSession
from .model import DeterministicEvaluator, load_checkpoint

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _runtime_asset(relative: str) -> Path:
    candidates = (
        PROJECT_ROOT / relative,
        Path.cwd() / relative,
        Path("/app") / relative,
    )
    return next((candidate for candidate in candidates if candidate.exists()), candidates[0])


DEFAULT_STATIC_DIR = _runtime_asset("static")
DEFAULT_CONFIG_PATH = _runtime_asset("configs/production.json")
POSITION_HASH_PREFIX = "gomoku-zero:position:v1|"
NDJSON_MEDIA_TYPE = "application/x-ndjson"
MAX_REQUEST_STAGES = 12


class AnalyzeRequest(BaseModel):
    """One immutable root position and optional cumulative search budgets."""

    model_config = ConfigDict(extra="forbid")

    board: list[StrictInt] = Field(min_length=BOARD_CELLS, max_length=BOARD_CELLS)
    budgets: list[StrictInt] | None = None
    mode: Literal["instant", "deep"] = "instant"

    @field_validator("board")
    @classmethod
    def validate_stones(cls, values: list[int]) -> list[int]:
        for index, value in enumerate(values):
            if value not in (WHITE, EMPTY, BLACK):
                raise ValueError(f"board[{index}] must be -1 (white), 0 (empty), or 1 (black)")
        return values

    @field_validator("budgets")
    @classmethod
    def validate_budgets(cls, values: list[int] | None) -> list[int] | None:
        if values is None:
            return None
        if not values:
            raise ValueError("budgets must contain at least one cumulative budget")
        if len(values) > MAX_REQUEST_STAGES:
            raise ValueError(f"budgets may contain at most {MAX_REQUEST_STAGES} stages")
        if any(value <= 0 for value in values):
            raise ValueError("budgets must contain positive integers")
        if values != sorted(set(values)):
            raise ValueError("budgets must be strictly increasing and unique")
        return values


@dataclass(frozen=True, slots=True)
class ServingSettings:
    """Small serving-only configuration, independent of training startup."""

    default_budgets: tuple[int, ...] = (40, 200, 1_000, 3_000, 6_000)
    max_simulations: int = 6_000
    max_concurrent_analyses: int = 1
    search_chunk_size: int = 256
    c_puct: float = 1.5
    seed: int = 0

    def __post_init__(self) -> None:
        if not self.default_budgets:
            raise ValueError("default_budgets cannot be empty")
        if tuple(sorted(set(self.default_budgets))) != self.default_budgets:
            raise ValueError("default_budgets must be strictly increasing and unique")
        if any(value <= 0 for value in self.default_budgets):
            raise ValueError("default_budgets must be positive")
        if self.max_simulations <= 0:
            raise ValueError("max_simulations must be positive")
        if self.default_budgets[-1] > self.max_simulations:
            raise ValueError("default budget exceeds max_simulations")
        if self.max_concurrent_analyses <= 0:
            raise ValueError("max_concurrent_analyses must be positive")
        if self.search_chunk_size <= 0:
            raise ValueError("search_chunk_size must be positive")
        if not math.isfinite(self.c_puct) or self.c_puct < 0.0:
            raise ValueError("c_puct must be finite and non-negative")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")

    @classmethod
    def from_environment(cls) -> ServingSettings:
        config_path = Path(
            os.getenv("GOMOKU_CONFIG") or os.getenv("GOMOKU_DEFAULT_CONFIG") or DEFAULT_CONFIG_PATH
        ).expanduser()
        configured_budgets = (40, 200, 1_000, 3_000, 6_000)
        configured_max = 6_000
        if config_path.is_file():
            try:
                raw = json.loads(config_path.read_text(encoding="utf-8"))
                serving = raw.get("serving", {}) if isinstance(raw, dict) else {}
                if isinstance(serving, dict):
                    configured_budgets = tuple(
                        int(value)
                        for value in serving.get("progressive_budgets", configured_budgets)
                    )
                    configured_max = int(serving.get("max_simulations", configured_max))
            except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid serving config {config_path.name}: {exc}") from exc

        max_simulations = _environment_int("GOMOKU_MAX_SIMULATIONS", configured_max, minimum=1)
        raw_budgets = os.getenv("GOMOKU_PROGRESSIVE_BUDGETS")
        if raw_budgets:
            try:
                configured_budgets = tuple(
                    int(value.strip()) for value in raw_budgets.split(",") if value.strip()
                )
            except ValueError as exc:
                raise ValueError(
                    "GOMOKU_PROGRESSIVE_BUDGETS must be comma-separated integers"
                ) from exc

        # A lower deployment cap safely trims config defaults instead of making
        # the service fail at import time. Explicit request budgets above it are
        # still rejected with a clear 422 response.
        budgets = tuple(value for value in configured_budgets if 0 < value <= max_simulations)
        if not budgets:
            budgets = (max_simulations,)
        return cls(
            default_budgets=budgets,
            max_simulations=max_simulations,
            max_concurrent_analyses=_environment_int(
                "GOMOKU_MAX_CONCURRENT_ANALYSES", 1, minimum=1
            ),
            search_chunk_size=_environment_int("GOMOKU_SEARCH_CHUNK_SIZE", 256, minimum=1),
            c_puct=_environment_float("GOMOKU_C_PUCT", 1.5, minimum=0.0),
            seed=_environment_int("GOMOKU_SERVING_SEED", 0, minimum=0),
        )


def _environment_int(name: str, default: int, *, minimum: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _environment_float(name: str, default: float, *, minimum: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(value) or value < minimum:
        raise ValueError(f"{name} must be finite and >= {minimum}")
    return value


class Evaluator(Protocol):
    def __call__(self, board: Board) -> tuple[np.ndarray, np.ndarray]: ...


class BootstrapHeuristicEvaluator:
    """Deterministic legal heuristic used only when no checkpoint exists.

    The policy favours immediate wins, mandatory blocks, connected lines, open
    ends, and central intersections. The three output scores are a normalized
    heuristic shape—not empirical win rates and not a trained prediction.
    """

    name = "deterministic-legal-heuristic-v1"
    _directions = ((0, 1), (1, 0), (1, 1), (1, -1))
    _run_weights = (0.0, 2.0, 11.0, 72.0, 720.0)

    def __call__(self, board: Board) -> tuple[np.ndarray, np.ndarray]:
        if board.is_terminal:
            return np.zeros(BOARD_CELLS, dtype=np.float64), board.wdl_target().astype(np.float64)

        legal = np.flatnonzero(board.legal_mask)
        policy_scores = np.zeros(BOARD_CELLS, dtype=np.float64)
        black_peak = 0.0
        white_peak = 0.0
        player = board.to_play

        for move_value in legal:
            move = int(move_value)
            black_strength = self._stone_strength(board.cells, move, BLACK)
            white_strength = self._stone_strength(board.cells, move, WHITE)
            black_peak = max(black_peak, black_strength)
            white_peak = max(white_peak, white_strength)
            own = black_strength if player == BLACK else white_strength
            block = white_strength if player == BLACK else black_strength
            row, col = divmod(move, BOARD_SIZE)
            centre_distance = abs(row - BOARD_SIZE // 2) + abs(col - BOARD_SIZE // 2)
            centre = (BOARD_SIZE - centre_distance) / BOARD_SIZE
            # A block matters almost as much as building the side-to-move's own
            # line. Every legal point receives positive mass.
            policy_scores[move] = 1.0 + own + 0.88 * block + 0.75 * max(0.0, centre)

        legal_scores = policy_scores[legal]
        total = float(legal_scores.sum())
        if not math.isfinite(total) or total <= 0.0:
            policy_scores[legal] = 1.0 / float(len(legal))
        else:
            policy_scores[legal] /= total

        wdl = self._heuristic_wdl(board, black_peak, white_peak)
        return policy_scores, wdl

    @classmethod
    def _stone_strength(cls, cells: Sequence[int], move: int, stone: int) -> float:
        row, col = divmod(move, BOARD_SIZE)
        best = 0.0
        total_score = 0.0
        for delta_row, delta_col in cls._directions:
            backward, backward_open = cls._ray(cells, row, col, -delta_row, -delta_col, stone)
            forward, forward_open = cls._ray(cells, row, col, delta_row, delta_col, stone)
            connected = backward + forward
            open_ends = int(backward_open) + int(forward_open)
            if connected >= 4:
                line_score = 12_000.0
            else:
                line_score = cls._run_weights[connected] * (1.0 + 0.32 * open_ends)
            total_score += line_score
            best = max(best, line_score)
        return best + 0.12 * total_score

    @staticmethod
    def _ray(
        cells: Sequence[int],
        row: int,
        col: int,
        delta_row: int,
        delta_col: int,
        stone: int,
    ) -> tuple[int, bool]:
        count = 0
        scan_row = row + delta_row
        scan_col = col + delta_col
        while (
            0 <= scan_row < BOARD_SIZE
            and 0 <= scan_col < BOARD_SIZE
            and cells[scan_row * BOARD_SIZE + scan_col] == stone
        ):
            count += 1
            scan_row += delta_row
            scan_col += delta_col
        open_end = (
            0 <= scan_row < BOARD_SIZE
            and 0 <= scan_col < BOARD_SIZE
            and cells[scan_row * BOARD_SIZE + scan_col] == EMPTY
        )
        return count, open_end

    @staticmethod
    def _heuristic_wdl(board: Board, black_peak: float, white_peak: float) -> np.ndarray:
        scale = max(72.0, black_peak + white_peak)
        line_balance = (black_peak - white_peak) / scale
        tempo = 0.08 if board.to_play == BLACK else -0.08
        log_odds = max(-4.0, min(4.0, 3.2 * line_balance + tempo))
        black_share = 1.0 / (1.0 + math.exp(-log_odds))
        threat = min(1.0, max(black_peak, white_peak) / 12_000.0)
        occupancy = board.move_count / BOARD_CELLS
        draw = max(0.04, min(0.22, 0.18 - 0.11 * threat + 0.04 * occupancy))
        decisive = 1.0 - draw
        return np.array(
            [decisive * black_share, draw, decisive * (1.0 - black_share)],
            dtype=np.float64,
        )


class _LockedEvaluator:
    """Serialize shared checkpoint inference across request worker threads."""

    def __init__(self, evaluator: DeterministicEvaluator) -> None:
        self._evaluator = evaluator
        self._lock = threading.Lock()

    def __call__(self, board: Board) -> tuple[np.ndarray, np.ndarray]:
        with self._lock:
            return self._evaluator(board)


class AnalysisRuntime:
    """Lazy evaluator plus public, path-safe model provenance."""

    def __init__(
        self,
        checkpoint_path: str | os.PathLike[str] | None = None,
        *,
        configured_but_missing: str | None = None,
        device: str = "cpu",
        source: str | None = None,
    ) -> None:
        self._checkpoint_path = (
            Path(checkpoint_path).expanduser().resolve() if checkpoint_path is not None else None
        )
        self._device = device
        self._unavailable_reason = configured_but_missing
        self._lock = threading.Lock()
        self._evaluator: Evaluator | None = None
        self._checkpoint_payload: dict[str, Any] | None = None

        if self._checkpoint_path is None and configured_but_missing is not None:
            self._info = {
                "status": "checkpoint-unavailable",
                "trained": False,
                "loaded": False,
                "model_id": "checkpoint-unavailable",
                "checkpoint": configured_but_missing,
                "checkpoint_sha256": None,
                "checkpoint_source": source,
                "evaluator": "unavailable",
                "device": device,
                "training_step": None,
                "warning": (
                    f"Configured checkpoint {configured_but_missing!r} is unavailable. "
                    "Analysis is disabled instead of silently falling back to an untrained model."
                ),
                "warning_zh": ("已配置的检查点不可用。服务已停止分析，不会静默回退到未训练模型。"),
            }
        elif self._checkpoint_path is None:
            warning = (
                "No trained checkpoint is loaded. Results come from a deterministic legal "
                "heuristic; they are not trained win rates or independent evaluation statistics."
            )
            self._info: dict[str, Any] = {
                "status": "bootstrap-untrained",
                "trained": False,
                "loaded": True,
                "model_id": "bootstrap-untrained-v1",
                "checkpoint": None,
                "checkpoint_sha256": None,
                "checkpoint_source": None,
                "evaluator": BootstrapHeuristicEvaluator.name,
                "device": "cpu",
                "training_step": None,
                "warning": warning,
                "warning_zh": (
                    "当前未加载训练检查点，结果来自确定性合法动作启发式；"
                    "它不是训练后的胜率，也不是独立评估统计。"
                ),
            }
        else:
            digest = _sha256_file(self._checkpoint_path)
            self._info = {
                "status": "checkpoint-configured",
                "trained": True,
                "loaded": False,
                "model_id": f"gomoku-checkpoint-{digest[:12]}",
                "checkpoint": self._checkpoint_path.name,
                "checkpoint_sha256": digest,
                "checkpoint_source": source or "local",
                "evaluator": "torch-policy-wdl-mcts",
                "device": device,
                "training_step": None,
                "warning": (
                    "Estimates reflect this checkpoint and the stated search budget; "
                    "they are not a mathematical game-theory result."
                ),
                "warning_zh": ("结果只代表该检查点与当前搜索预算下的判断，不是五子棋的数学真值。"),
            }

    @classmethod
    def from_environment(cls) -> AnalysisRuntime:
        configured = os.getenv("GOMOKU_CHECKPOINT")
        configured_uri = os.getenv("GOMOKU_CHECKPOINT_URI")
        device = os.getenv("GOMOKU_DEVICE", "cpu")
        if configured:
            path = _resolve_checkpoint_candidate(Path(configured).expanduser())
            if path is None:
                return cls(
                    configured_but_missing=Path(configured).name,
                    device=device,
                    source="local",
                )
            return cls(path, device=device, source="local")

        if configured_uri:
            checkpoint_name = _checkpoint_uri_name(configured_uri)
            try:
                path = _materialize_checkpoint_uri(configured_uri)
            except Exception:
                LOGGER.exception("failed to materialize configured checkpoint URI")
                return cls(
                    configured_but_missing=checkpoint_name,
                    device=device,
                    source="gcs" if configured_uri.startswith("gs://") else "uri",
                )
            return cls(
                path,
                device=device,
                source="gcs" if configured_uri.startswith("gs://") else "uri",
            )

        candidates = (
            _runtime_asset("checkpoints/best.pt"),
            _runtime_asset("checkpoints/latest.pt"),
            _runtime_asset("artifacts/checkpoints/best.pt"),
        )
        for candidate in candidates:
            if candidate.is_file():
                return cls(candidate, device=device)
        return cls(device=device)

    def evaluator(self) -> Evaluator:
        with self._lock:
            if self._evaluator is not None:
                return self._evaluator
            if self._unavailable_reason is not None:
                raise RuntimeError("configured checkpoint is unavailable")
            if self._checkpoint_path is None:
                self._evaluator = BootstrapHeuristicEvaluator()
                return self._evaluator
            try:
                model, payload = load_checkpoint(
                    self._checkpoint_path,
                    map_location=self._device,
                )
                self._checkpoint_payload = payload
                self._evaluator = _LockedEvaluator(
                    DeterministicEvaluator(model, device=self._device)
                )
                self._info["status"] = "checkpoint-loaded"
                self._info["loaded"] = True
                step = payload.get("step")
                self._info["training_step"] = step if isinstance(step, int) else None
                return self._evaluator
            except Exception:
                self._info["status"] = "checkpoint-error"
                self._info["loaded"] = False
                LOGGER.exception("failed to load configured Gomoku checkpoint")
                raise

    def info(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._info)

    def warmup(self) -> None:
        """Strict-load and execute one real inference for readiness."""

        try:
            evaluator = self.evaluator()
            evaluator(Board.initial())
        except Exception:
            with self._lock:
                if self._checkpoint_path is not None:
                    self._info["status"] = "checkpoint-error"
                    self._info["loaded"] = False
            raise

    @property
    def estimate_kind(self) -> str:
        if self._unavailable_reason is not None:
            return "checkpoint-unavailable"
        return "checkpoint-mcts" if self._checkpoint_path is not None else "deterministic-heuristic"

    @property
    def has_configured_checkpoint(self) -> bool:
        return self._checkpoint_path is not None or self._unavailable_reason is not None


def _resolve_checkpoint_candidate(path: Path) -> Path | None:
    if path.is_file():
        return path.resolve()
    if path.is_dir():
        for name in ("best.pt", "latest.pt"):
            candidate = path / name
            if candidate.is_file():
                return candidate.resolve()
        checkpoints = sorted(path.glob("*.pt"), key=lambda item: item.stat().st_mtime, reverse=True)
        if checkpoints:
            return checkpoints[0].resolve()
    return None


def _checkpoint_uri_name(uri: str) -> str:
    parsed = urlparse(uri)
    name = Path(parsed.path).name
    return name or "configured-checkpoint.pt"


def _materialize_checkpoint_uri(uri: str) -> Path:
    """Download a configured checkpoint to an isolated, atomic local file."""

    parsed = urlparse(uri)
    if parsed.scheme == "file":
        candidate = _resolve_checkpoint_candidate(Path(parsed.path))
        if candidate is None:
            raise FileNotFoundError(f"checkpoint URI does not exist: {_checkpoint_uri_name(uri)}")
        return candidate
    if parsed.scheme != "gs":
        raise ValueError("GOMOKU_CHECKPOINT_URI must use gs:// or file://")
    bucket_name = parsed.netloc
    object_name = parsed.path.lstrip("/")
    if not bucket_name or not object_name:
        raise ValueError("GCS checkpoint URI must contain a bucket and object name")

    # Hash the URI into the directory rather than the filename. This prevents
    # collisions while keeping public provenance limited to the object basename.
    uri_digest = hashlib.sha256(uri.encode("utf-8")).hexdigest()[:16]
    cache_dir = Path(tempfile.gettempdir()) / "gomoku-zero-checkpoints" / uri_digest
    cache_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    target = cache_dir / _checkpoint_uri_name(uri)
    if target.is_file() and target.stat().st_size > 0:
        return target.resolve()

    try:
        from google.cloud import storage
    except ImportError as exc:  # pragma: no cover - deployment image includes the extra.
        raise RuntimeError("google-cloud-storage is required for gs:// checkpoints") from exc

    temporary = cache_dir / f".{target.name}.{uuid4().hex}.part"
    try:
        client = storage.Client()
        blob = client.bucket(bucket_name).blob(object_name)
        blob.download_to_filename(str(temporary))
        if not temporary.is_file() or temporary.stat().st_size <= 0:
            raise ValueError("downloaded checkpoint is empty")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target.resolve()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def derive_position_hash(cells: Sequence[int], to_play: int) -> str:
    canonical = f"{POSITION_HASH_PREFIX}{','.join(str(value) for value in cells)}|{to_play}"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _validated_board(values: Sequence[int]) -> Board:
    black_count = sum(value == BLACK for value in values)
    white_count = sum(value == WHITE for value in values)
    if black_count == white_count:
        to_play = BLACK
    elif black_count == white_count + 1:
        to_play = WHITE
    else:
        raise HTTPException(
            status_code=422,
            detail=(
                "invalid alternating stone counts: black must equal white or exceed white by one"
            ),
        )
    board = Board.from_flat(values, to_play=to_play)
    if board.is_terminal:
        raise HTTPException(
            status_code=422,
            detail="the supplied position is terminal and has no legal analysis moves",
        )
    if not board.legal_moves:
        raise HTTPException(status_code=422, detail="the supplied position has no legal moves")
    return board


def _request_budgets(payload: AnalyzeRequest, settings: ServingSettings) -> tuple[int, ...]:
    if payload.budgets is None:
        budgets = (
            (settings.default_budgets[0],)
            if payload.mode == "instant"
            else settings.default_budgets
        )
    else:
        budgets = tuple(payload.budgets)
    if budgets[-1] > settings.max_simulations:
        raise HTTPException(
            status_code=422,
            detail=(
                f"requested budget {budgets[-1]} exceeds the server cap "
                f"of {settings.max_simulations} simulations"
            ),
        )
    return budgets


def _normalized_outcome(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size != 3:
        raise ValueError("root WDL must contain exactly black, draw, and white")
    array = np.where(np.isfinite(array) & (array >= 0.0), array, 0.0)
    total = float(array.sum())
    if not math.isfinite(total) or total <= 0.0:
        array = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    else:
        array /= total
    return {"black": float(array[0]), "draw": float(array[1]), "white": float(array[2])}


def _finite_float(value: float, *, lower: float, upper: float) -> float:
    number = float(value)
    if not math.isfinite(number):
        return 0.0
    return max(lower, min(upper, number))


def _top_legal_moves(result: SearchResult, board: Board) -> list[dict[str, int | float | str]]:
    legal = set(board.legal_moves)
    ranked = sorted(result.moves, key=lambda item: (-item.visits, -item.q_value, item.move))
    top: list[dict[str, int | float | str]] = []
    seen: set[int] = set()
    for item in ranked:
        if item.move not in legal or board.cells[item.move] != EMPTY or item.move in seen:
            continue
        seen.add(item.move)
        visit_share = (
            float(result.visit_policy[item.move])
            if 0 <= item.move < len(result.visit_policy)
            else item.visit_fraction
        )
        top.append(
            {
                "move": int(item.move),
                "row": int(item.row),
                "col": int(item.col),
                "coordinate": item.coordinate,
                "visits": int(item.visits),
                "visit_share": _finite_float(visit_share, lower=0.0, upper=1.0),
                "prior": _finite_float(item.network_prior, lower=0.0, upper=1.0),
                "q_value": _finite_float(item.q_value, lower=-1.0, upper=1.0),
            }
        )
        if len(top) == 3:
            break
    return top


def _analysis_frame(
    *,
    result: SearchResult,
    board: Board,
    position_hash: str,
    request_id: str,
    stage: int,
    stages: int,
    target: int,
    runtime: AnalysisRuntime,
) -> dict[str, Any]:
    return {
        "type": "analysis",
        "schema_version": 1,
        "request_id": request_id,
        "analysis_id": request_id,
        "sequence": stage,
        "position_hash": position_hash,
        "stage": stage,
        "stages": stages,
        "target_simulations": target,
        "simulations": int(result.simulations),
        "complete": stage == stages,
        "to_play": int(board.to_play),
        "to_play_name": "black" if board.to_play == BLACK else "white",
        "legal_moves": len(board.legal_moves),
        "top_moves": _top_legal_moves(result, board),
        "outcome": _normalized_outcome(result.root_wdl),
        "root_value": _finite_float(result.root_value, lower=-1.0, upper=1.0),
        "estimate_kind": runtime.estimate_kind,
        "model": runtime.info(),
    }


def _ndjson(payload: dict[str, Any]) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n"
    ).encode("utf-8")


SessionFactory = Callable[[Board, Evaluator, MCTSConfig], SearchSession]


def create_app(
    *,
    settings: ServingSettings | None = None,
    runtime: AnalysisRuntime | None = None,
    session_factory: SessionFactory = SearchSession,
    static_dir: str | os.PathLike[str] | None = None,
) -> FastAPI:
    """Create an independently testable ASGI application."""

    serving = settings or ServingSettings.from_environment()
    model_runtime = runtime or AnalysisRuntime.from_environment()
    assets = Path(static_dir or os.getenv("GOMOKU_STATIC_DIR", DEFAULT_STATIC_DIR)).resolve()

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        # Explicitly configured trained deployments are strict-loaded and run a
        # real empty-board inference before readiness can report success.
        if bool(getattr(model_runtime, "has_configured_checkpoint", False)):
            try:
                warmup = getattr(model_runtime, "warmup", None)
                if callable(warmup):
                    await asyncio.to_thread(warmup)
                else:
                    loaded_evaluator = await asyncio.to_thread(model_runtime.evaluator)
                    await asyncio.to_thread(loaded_evaluator, Board.initial())
            except Exception:
                LOGGER.exception("checkpoint warmup failed; refusing to start this revision")
                raise
        yield

    application = FastAPI(
        title="Gomoku Zero Analysis API",
        version=__version__,
        summary="Progressive, legality-safe PUCT analysis for 15x15 Gomoku",
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
        lifespan=lifespan,
    )
    application.state.analysis_semaphore = asyncio.Semaphore(serving.max_concurrent_analyses)
    application.state.settings = serving
    application.state.runtime = model_runtime

    cors_origins = [
        origin.strip()
        for origin in os.getenv("GOMOKU_CORS_ORIGINS", "").split(",")
        if origin.strip()
    ]
    if cors_origins:
        application.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["Accept", "Content-Type"],
        )

    @application.middleware("http")
    async def security_headers(request: Request, call_next: Callable[..., Any]) -> Any:
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
        response.headers.setdefault(
            "Content-Security-Policy",
            "default-src 'self'; base-uri 'self'; connect-src 'self'; "
            "font-src 'self'; frame-ancestors 'none'; img-src 'self' data:; "
            "object-src 'none'; script-src 'self'; style-src 'self'",
        )
        return response

    @application.get("/api/health", tags=["service"])
    async def health() -> dict[str, Any]:
        info = model_runtime.info()
        ready = info["status"] in {"bootstrap-untrained", "checkpoint-loaded"}
        return {
            "status": "ok" if ready else "degraded",
            "service": "gomoku-zero",
            "version": __version__,
            "board_size": BOARD_SIZE,
            "board_cells": BOARD_CELLS,
            "ready": ready,
            "model": info,
            "default_budgets": list(serving.default_budgets),
            "max_simulations": serving.max_simulations,
            "max_concurrent_analyses": serving.max_concurrent_analyses,
            "search_chunk_size": serving.search_chunk_size,
            "position_hash": "sha256(gomoku-zero:position:v1|<comma-cells>|<to-play>)",
        }

    @application.get("/api/model", tags=["service"])
    async def model_provenance() -> dict[str, Any]:
        return model_runtime.info()

    @application.post(
        "/api/analyze",
        tags=["analysis"],
        responses={
            200: {
                "description": "One newline-delimited JSON frame per cumulative budget",
                "content": {
                    NDJSON_MEDIA_TYPE: {
                        "schema": {"type": "string", "format": "binary"},
                        "example": (
                            '{"type":"analysis","position_hash":"...",'
                            '"simulations":40,"complete":false,"top_moves":[]}\n'
                        ),
                    }
                },
            },
            422: {"description": "Invalid board or search budgets"},
            503: {"description": "Configured checkpoint could not be loaded"},
        },
    )
    async def analyze(payload: AnalyzeRequest, request: Request) -> StreamingResponse:
        board = _validated_board(payload.board)
        budgets = _request_budgets(payload, serving)
        position_hash = derive_position_hash(board.cells, board.to_play)
        request_id = uuid4().hex

        try:
            evaluator = await asyncio.to_thread(model_runtime.evaluator)
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail="the configured model checkpoint could not be loaded",
            ) from exc

        async def stream() -> AsyncIterator[bytes]:
            try:
                async with application.state.analysis_semaphore:
                    if await request.is_disconnected():
                        return
                    config = MCTSConfig(
                        c_puct=serving.c_puct,
                        add_root_noise=False,
                        seed=serving.seed,
                    )
                    # Exactly one tree is constructed per request and reused by
                    # all cumulative run_until calls below.
                    session = await asyncio.to_thread(
                        session_factory,
                        board,
                        evaluator,
                        config,
                    )
                    completed_simulations = 0
                    for index, target in enumerate(budgets, start=1):
                        result: SearchResult | None = None
                        while completed_simulations < target:
                            if await request.is_disconnected():
                                return
                            chunk_target = min(
                                target,
                                completed_simulations + serving.search_chunk_size,
                            )
                            result = await asyncio.to_thread(session.run_until, chunk_target)
                            completed_simulations = chunk_target
                        if result is None:
                            raise RuntimeError("search stage did not advance")
                        yield _ndjson(
                            _analysis_frame(
                                result=result,
                                board=board,
                                position_hash=position_hash,
                                request_id=request_id,
                                stage=index,
                                stages=len(budgets),
                                target=target,
                                runtime=model_runtime,
                            )
                        )
                        await asyncio.sleep(0)
            except asyncio.CancelledError:
                raise
            except Exception:
                LOGGER.exception("analysis request %s failed", request_id)
                yield _ndjson(
                    {
                        "type": "error",
                        "schema_version": 1,
                        "request_id": request_id,
                        "analysis_id": request_id,
                        "position_hash": position_hash,
                        "error": {
                            "code": "analysis_failed",
                            "message": "analysis could not be completed",
                        },
                    }
                )

        return StreamingResponse(
            stream(),
            media_type=NDJSON_MEDIA_TYPE,
            headers={
                "Cache-Control": "no-store, no-transform",
                "X-Accel-Buffering": "no",
                "X-Position-Hash": position_hash,
                "X-Request-ID": request_id,
            },
        )

    @application.get("/", include_in_schema=False, response_model=None)
    async def index() -> Response:
        index_path = assets / "index.html"
        if not index_path.is_file():
            return JSONResponse(
                status_code=503,
                content={"detail": "web assets are not installed", "api_docs": "/api/docs"},
            )
        return FileResponse(
            index_path,
            media_type="text/html",
            headers={"Cache-Control": "no-cache"},
        )

    if assets.is_dir():
        application.mount("/static", StaticFiles(directory=assets), name="static")

    return application


app = create_app()


def main() -> None:
    """Run the public service through the ``gomoku-api`` console script."""

    host = os.getenv("HOST", "0.0.0.0")
    port = _environment_int("PORT", 8080, minimum=1)
    uvicorn.run("gomoku_zero.api:app", host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
