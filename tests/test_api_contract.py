from __future__ import annotations

import hashlib
import json
import pickle
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from gomoku_zero.api import (
    AnalysisRuntime,
    ServingSettings,
    create_app,
    derive_position_hash,
)
from gomoku_zero.game import BLACK, BOARD_CELLS, EMPTY, WHITE, Board
from gomoku_zero.mcts import MCTSConfig, SearchSession

TEST_SETTINGS = ServingSettings(
    default_budgets=(1, 2, 4),
    max_simulations=4,
    max_concurrent_analyses=1,
    c_puct=1.5,
    seed=11,
)


def _frames(response: Any) -> list[dict[str, Any]]:
    return [json.loads(line) for line in response.text.splitlines() if line.strip()]


def _client(*, session_factory: Callable[..., Any] = SearchSession) -> TestClient:
    return TestClient(
        create_app(
            settings=TEST_SETTINGS,
            runtime=AnalysisRuntime(),
            session_factory=session_factory,
        )
    )


def test_health_and_model_report_bootstrap_provenance() -> None:
    with _client() as client:
        health = client.get("/api/health")
        model = client.get("/api/model")

    assert health.status_code == 200
    assert health.headers["content-type"].startswith("application/json")
    body = health.json()
    assert body["status"] == "ok"
    assert body["service"] == "gomoku-zero"
    assert body["board_size"] == 15
    assert body["board_cells"] == 225
    assert body["default_budgets"] == [1, 2, 4]
    assert body["max_simulations"] == 4
    assert body["model"]["status"] == "bootstrap-untrained"

    assert model.status_code == 200
    provenance = model.json()
    assert provenance["status"] == "bootstrap-untrained"
    assert provenance["trained"] is False
    assert provenance["model_id"] == "bootstrap-untrained-v1"
    assert provenance["checkpoint"] is None
    assert provenance["checkpoint_sha256"] is None
    assert provenance["evaluator"] == "deterministic-legal-heuristic-v1"
    assert "not trained win rates" in provenance["warning"]


def test_progressive_stream_is_cumulative_and_reuses_one_tree() -> None:
    sessions: list[Any] = []

    class RecordingSession:
        def __init__(self, board: Board, evaluator: Any, config: MCTSConfig) -> None:
            self.inner = SearchSession(board, evaluator, config)
            self.calls: list[int] = []
            sessions.append(self)

        def run_until(self, target: int) -> Any:
            self.calls.append(target)
            return self.inner.run_until(target)

    with _client(session_factory=RecordingSession) as client:
        response = client.post(
            "/api/analyze",
            json={"board": [EMPTY] * BOARD_CELLS, "budgets": [1, 2, 4]},
            headers={"Accept": "application/x-ndjson"},
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/x-ndjson")
    frames = _frames(response)
    assert len(frames) == 3
    assert [frame["simulations"] for frame in frames] == [1, 2, 4]
    assert [frame["target_simulations"] for frame in frames] == [1, 2, 4]
    assert [frame["stage"] for frame in frames] == [1, 2, 3]
    assert [frame["sequence"] for frame in frames] == [1, 2, 3]
    assert [frame["complete"] for frame in frames] == [False, False, True]
    assert len({frame["analysis_id"] for frame in frames}) == 1
    assert len({frame["position_hash"] for frame in frames}) == 1
    assert response.headers["x-position-hash"] == frames[0]["position_hash"]
    assert frames[0]["position_hash"] == derive_position_hash([EMPTY] * BOARD_CELLS, BLACK)
    assert len(sessions) == 1
    assert sessions[0].calls == [1, 2, 4]

    for frame in frames:
        assert frame["to_play"] == BLACK
        assert frame["legal_moves"] == BOARD_CELLS
        assert frame["estimate_kind"] == "deterministic-heuristic"
        assert len(frame["top_moves"]) == 3
        assert len({move["move"] for move in frame["top_moves"]}) == 3
        for move in frame["top_moves"]:
            assert 0 <= move["move"] < BOARD_CELLS
            assert move["row"] == move["move"] // 15
            assert move["col"] == move["move"] % 15
            assert move["visits"] >= 0
            assert 0.0 <= move["visit_share"] <= 1.0
            assert 0.0 <= move["prior"] <= 1.0
            assert -1.0 <= move["q_value"] <= 1.0
        outcome = frame["outcome"]
        assert set(outcome) == {"black", "draw", "white"}
        assert sum(outcome.values()) == pytest.approx(1.0)
        assert all(0.0 <= value <= 1.0 for value in outcome.values())


def test_large_stage_is_chunked_without_emitting_intermediate_frames() -> None:
    calls: list[int] = []

    class ChunkRecordingSession:
        def __init__(self, board: Board, evaluator: Any, config: MCTSConfig) -> None:
            self.inner = SearchSession(board, evaluator, config)

        def run_until(self, target: int) -> Any:
            calls.append(target)
            return self.inner.run_until(target)

    settings = ServingSettings(
        default_budgets=(1, 5),
        max_simulations=5,
        max_concurrent_analyses=1,
        search_chunk_size=2,
    )
    app = create_app(
        settings=settings,
        runtime=AnalysisRuntime(),
        session_factory=ChunkRecordingSession,
    )
    with TestClient(app) as client:
        response = client.post(
            "/api/analyze",
            json={"board": [EMPTY] * BOARD_CELLS, "mode": "deep"},
        )

    assert response.status_code == 200
    assert calls == [1, 3, 5]
    assert [frame["simulations"] for frame in _frames(response)] == [1, 5]


def test_occupied_high_policy_action_can_never_be_recommended() -> None:
    occupied = 112  # H8

    def adversarial_evaluator(board: Board) -> tuple[np.ndarray, np.ndarray]:
        policy = np.ones(BOARD_CELLS, dtype=np.float64)
        policy[occupied] = 1e30
        return policy, np.array([0.3, 0.2, 0.5], dtype=np.float64)

    class AdversarialRuntime:
        estimate_kind = "deterministic-heuristic"

        @staticmethod
        def evaluator() -> Any:
            return adversarial_evaluator

        @staticmethod
        def info() -> dict[str, Any]:
            return {
                "status": "bootstrap-untrained",
                "trained": False,
                "loaded": True,
                "model_id": "test-adversarial",
                "checkpoint": None,
                "checkpoint_sha256": None,
                "evaluator": "test-adversarial",
                "device": "cpu",
                "training_step": None,
                "warning": "test evaluator",
            }

    app = create_app(settings=TEST_SETTINGS, runtime=AdversarialRuntime())  # type: ignore[arg-type]
    board = [EMPTY] * BOARD_CELLS
    board[occupied] = BLACK
    with TestClient(app) as client:
        response = client.post("/api/analyze", json={"board": board, "budgets": [1, 2, 4]})

    assert response.status_code == 200
    for frame in _frames(response):
        assert frame["legal_moves"] == BOARD_CELLS - 1
        assert frame["to_play"] == WHITE
        recommended = {item["move"] for item in frame["top_moves"]}
        assert occupied not in recommended
        assert all(board[move] == EMPTY for move in recommended)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"board": [EMPTY] * (BOARD_CELLS - 1)}, "225"),
        ({"board": [EMPTY] * (BOARD_CELLS - 1) + [2]}, "-1"),
        ({"board": [EMPTY] * (BOARD_CELLS - 1) + [True]}, "integer"),
        ({"board": [WHITE] + [EMPTY] * (BOARD_CELLS - 1)}, "stone counts"),
        ({"board": [EMPTY] * BOARD_CELLS, "budgets": []}, "at least one"),
        ({"board": [EMPTY] * BOARD_CELLS, "budgets": [2, 2]}, "strictly increasing"),
        ({"board": [EMPTY] * BOARD_CELLS, "budgets": [4, 2]}, "strictly increasing"),
        ({"board": [EMPTY] * BOARD_CELLS, "budgets": [5]}, "server cap"),
        ({"board": [EMPTY] * BOARD_CELLS, "unexpected": 1}, "Extra inputs"),
    ],
)
def test_invalid_requests_are_rejected(payload: dict[str, Any], message: str) -> None:
    with _client() as client:
        response = client.post("/api/analyze", json=payload)

    assert response.status_code == 422
    assert message.lower() in response.text.lower()


def test_terminal_board_is_rejected_before_search() -> None:
    board = [EMPTY] * BOARD_CELLS
    board[0:5] = [BLACK] * 5
    board[15:19] = [WHITE] * 4
    with _client() as client:
        response = client.post("/api/analyze", json={"board": board, "budgets": [1]})

    assert response.status_code == 422
    assert "terminal" in response.text


def test_instant_mode_uses_only_first_default_budget() -> None:
    with _client() as client:
        response = client.post(
            "/api/analyze",
            json={"board": [EMPTY] * BOARD_CELLS, "mode": "instant"},
        )

    frames = _frames(response)
    assert response.status_code == 200
    assert [frame["simulations"] for frame in frames] == [1]
    assert frames[0]["complete"] is True


def test_corrupt_gcs_checkpoint_prevents_revision_startup_without_bootstrap(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"not a torch checkpoint")
    monkeypatch.delenv("GOMOKU_CHECKPOINT", raising=False)
    monkeypatch.setenv("GOMOKU_CHECKPOINT_URI", "gs://model-bucket/releases/best.pt")
    monkeypatch.setattr(
        "gomoku_zero.api._materialize_checkpoint_uri",
        lambda _: checkpoint,
    )
    runtime = AnalysisRuntime.from_environment()
    app = create_app(settings=TEST_SETTINGS, runtime=runtime)

    with pytest.raises(pickle.UnpicklingError), TestClient(app):
        pass

    info = runtime.info()
    assert info["status"] == "checkpoint-error"
    assert info["loaded"] is False
    assert info["checkpoint_source"] == "gcs"
    assert info["checkpoint_sha256"] == derive_file_sha(checkpoint)


def derive_file_sha(path: Any) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_openapi_describes_ndjson_response() -> None:
    with _client() as client:
        schema = client.get("/api/openapi.json").json()

    response_content = schema["paths"]["/api/analyze"]["post"]["responses"]["200"]["content"]
    assert "application/x-ndjson" in response_content
