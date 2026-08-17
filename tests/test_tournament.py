from __future__ import annotations

from collections import Counter

from gomoku_zero.game import BLACK, WHITE
from gomoku_zero.model import GomokuNet
from gomoku_zero.tournament import (
    TournamentGame,
    build_round_robin_schedule,
    fit_elo,
    play_round_robin,
)


def _game(
    game_id: int,
    black: str,
    white: str,
    outcome: int,
    *,
    matchup: str,
    opening_pair: int,
) -> TournamentGame:
    return TournamentGame(
        game_id=game_id,
        matchup=matchup,
        opening_pair=opening_pair,
        seed=opening_pair,
        black=black,
        white=white,
        opening=(),
        moves=(),
        outcome=outcome,
    )


def test_round_robin_randomizes_first_color_and_balances_every_pair() -> None:
    schedule = build_round_robin_schedule(
        ("v0", "v1", "v2"),
        games_per_pair=6,
        opening_plies=4,
        seed=41,
    )
    assert len(schedule) == 18
    by_matchup: dict[str, list] = {}
    for game in schedule:
        by_matchup.setdefault(game.matchup, []).append(game)
    assert len(by_matchup) == 3
    for games in by_matchup.values():
        black_counts = Counter(game.black for game in games)
        assert sorted(black_counts.values()) == [3, 3]
        for opening_pair in range(3):
            paired = [game for game in games if game.opening_pair == opening_pair]
            assert len(paired) == 2
            assert paired[0].opening == paired[1].opening
            assert paired[0].black == paired[1].white
            assert paired[0].white == paired[1].black


def test_regularized_elo_orders_synthetic_strength() -> None:
    games: list[TournamentGame] = []
    game_id = 0
    for matchup, stronger, weaker in (
        ("strong__vs__base", "strong", "base"),
        ("base__vs__weak", "base", "weak"),
        ("strong__vs__weak", "strong", "weak"),
    ):
        for opening_pair in range(4):
            games.append(
                _game(
                    game_id,
                    stronger,
                    weaker,
                    BLACK,
                    matchup=matchup,
                    opening_pair=opening_pair,
                )
            )
            game_id += 1
            games.append(
                _game(
                    game_id,
                    weaker,
                    stronger,
                    WHITE,
                    matchup=matchup,
                    opening_pair=opening_pair,
                )
            )
            game_id += 1
    report = fit_elo(
        ("base", "strong", "weak"),
        games,
        anchor="base",
        bootstrap_samples=20,
        seed=7,
    )
    ratings = {row["model"]: row["elo"] for row in report["ratings"]}
    assert ratings["strong"] > ratings["base"] > ratings["weak"]
    assert ratings["base"] == 1000.0


def test_round_robin_uses_batched_search_and_returns_legal_games() -> None:
    model_a = GomokuNet(channels=4, residual_blocks=0)
    model_b = GomokuNet(channels=4, residual_blocks=0)
    games, telemetry = play_round_robin(
        {"a": model_a, "b": model_b},
        games_per_pair=2,
        simulations=1,
        opening_plies=4,
        seed=5,
        device="cpu",
        max_batch_size=2,
    )
    assert len(games) == 2
    assert {game.outcome for game in games} <= {BLACK, 0, WHITE}
    assert all(len(game.moves) >= 5 for game in games)
    assert telemetry["games"] == 2
    assert telemetry["models"]["a"]["inference_calls"] > 0
    assert telemetry["models"]["b"]["inference_calls"] > 0
