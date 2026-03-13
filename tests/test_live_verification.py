from __future__ import annotations

import tempfile
import unittest
from datetime import date
from pathlib import Path

from src.parlay.engine import Leg
from src.utils.config import normalize_lookup_key
from src.verification.nba_live import NbaLiveClient
from src.verification.roster_cache import LiveCache
from src.verification.verify_pick import verify_leg


class FakeLiveClient:
    def __init__(self, team_abbrev: str = "GSW", team_name: str = "Golden State Warriors", game_ok: bool = True):
        self.team_abbrev = team_abbrev
        self.team_name = team_name
        self.game_ok = game_ok

    def get_current_team(self, player_name: str) -> dict:
        return {
            "found": True,
            "player_name": "Jimmy Butler",
            "player_id": 202710,
            "current_team_abbrev": self.team_abbrev,
            "current_team": self.team_name,
            "verified_source": "nba_api",
        }

    def validate_game(self, home_team: str, away_team: str, game_date: str | None) -> dict:
        if not self.game_ok:
            return {
                "ok": False,
                "game_exists": False,
                "reason": "game_not_found",
                "home_team_abbrev": home_team,
                "away_team_abbrev": away_team,
                "game_date": game_date,
            }
        return {
            "ok": True,
            "game_exists": True,
            "home_team_abbrev": home_team,
            "away_team_abbrev": away_team,
            "game_date": game_date,
        }


class LiveVerificationTests(unittest.TestCase):
    def test_jimmy_butler_current_team_resolves_from_live_roster(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            client = NbaLiveClient(cache=LiveCache(Path(tmpdir), ttl_hours=12))
            client.get_rosters = lambda: {
                "players": {
                    normalize_lookup_key("Jimmy Butler"): {
                        "player_id": 202710,
                        "player_name": "Jimmy Butler",
                        "team_abbrev": "GSW",
                        "team_name": "Golden State Warriors",
                    }
                },
                "teams": {},
            }
            client.get_players = lambda: {"players": []}

            result = client.get_current_team("Butler")

            self.assertTrue(result["found"])
            self.assertEqual(result["player_name"], "Jimmy Butler")
            self.assertEqual(result["current_team_abbrev"], "GSW")

    def test_player_team_mismatch_rejects_leg(self) -> None:
        leg = Leg(
            leg_type="player_prop",
            description="Jimmy Butler Over 34.5 PRA",
            player="Jimmy Butler",
            stat_type="pra",
            line=34.5,
            team="MIA",
            opponent="NYK",
            game_date=date(2026, 3, 12),
        )

        result = verify_leg(
            leg,
            game_context={"home_team": "MIA", "away_team": "NYK", "game_date": "2026-03-12"},
            client=FakeLiveClient(team_abbrev="GSW"),
        )

        self.assertFalse(result.ok)
        self.assertEqual(result.reason, "team_mismatch")

    def test_valid_player_team_game_passes_verification(self) -> None:
        leg = Leg(
            leg_type="player_prop",
            description="Jimmy Butler Over 34.5 PRA",
            player="Jimmy Butler",
            stat_type="pra",
            line=34.5,
            team="GSW",
            opponent="NYK",
            game_date=date(2026, 3, 12),
        )

        result = verify_leg(
            leg,
            game_context={"home_team": "GSW", "away_team": "NYK", "game_date": "2026-03-12"},
            client=FakeLiveClient(team_abbrev="GSW"),
        )

        self.assertTrue(result.ok)
        self.assertTrue(result.metadata["verified"])
        self.assertEqual(result.corrected_leg.team, "GSW")

    def test_live_team_correction_applies_when_player_is_in_requested_game(self) -> None:
        leg = Leg(
            leg_type="player_prop",
            description="Jimmy Butler Over 34.5 PRA",
            player="Jimmy Butler",
            stat_type="pra",
            line=34.5,
            team="MIA",
            opponent="NYK",
            game_date=date(2026, 3, 12),
        )

        result = verify_leg(
            leg,
            game_context={"home_team": "GSW", "away_team": "NYK", "game_date": "2026-03-12"},
            client=FakeLiveClient(team_abbrev="GSW"),
        )

        self.assertTrue(result.ok)
        self.assertEqual(result.corrected_leg.team, "GSW")
        self.assertEqual(result.corrected_leg.opponent, "NYK")
        self.assertIn("corrected_team_from_live_data", result.warnings)


class LiveCacheTests(unittest.TestCase):
    def test_stale_cache_refresh_uses_loader(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = LiveCache(Path(tmpdir), ttl_hours=0)
            state = {"value": 0}

            def loader() -> dict:
                state["value"] += 1
                return {"value": state["value"]}

            first = cache.get_or_refresh("players", loader, ttl_hours=0)
            second = cache.get_or_refresh("players", loader, ttl_hours=0)

            self.assertEqual(first.payload["value"], 1)
            self.assertEqual(second.payload["value"], 2)
            self.assertTrue(second.refreshed)

    def test_stale_cache_fallback_is_used_when_loader_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = LiveCache(Path(tmpdir), ttl_hours=0)
            cache.save("rosters", {"value": "cached"}, ttl_hours=0)

            result = cache.get_or_refresh(
                "rosters",
                lambda: (_ for _ in ()).throw(RuntimeError("api down")),
                ttl_hours=0,
            )

            self.assertEqual(result.payload["value"], "cached")
            self.assertEqual(result.source, "stale-cache")
            self.assertTrue(result.stale)


if __name__ == "__main__":
    unittest.main()