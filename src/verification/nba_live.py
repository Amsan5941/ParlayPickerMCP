"""Live NBA roster and schedule verification backed by nba_api."""

from __future__ import annotations

import time
from functools import lru_cache
from typing import Any

from nba_api.stats.endpoints import commonteamroster, scoreboardv2
from nba_api.stats.static import players as static_players
from nba_api.stats.static import teams as static_teams

from src.utils.config import (
    LIVE_API_RETRIES,
    LIVE_REQUEST_TIMEOUT,
    LIVE_SEASON,
    VERIFY_SCHEDULES,
    normalize_lookup_key,
    resolve_team,
    resolve_team_name,
)
from src.utils.logger import get_logger
from src.verification.roster_cache import LiveCache

log = get_logger(__name__)


class NbaLiveClient:
    """Provides cached current-team and schedule verification via nba_api."""

    def __init__(
        self,
        cache: LiveCache | None = None,
        season: str | None = None,
        timeout: int = LIVE_REQUEST_TIMEOUT,
        retries: int = LIVE_API_RETRIES,
    ):
        self.cache = cache or LiveCache()
        self.season = season or LIVE_SEASON
        self.timeout = timeout
        self.retries = retries
        self._teams = static_teams.get_teams()
        self._team_by_id = {int(team["id"]): team for team in self._teams}
        self._team_by_abbrev = {team["abbreviation"]: team for team in self._teams}

    def _retry(self, fn):
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                return fn()
            except Exception as exc:  # pragma: no cover - network errors mocked in tests
                last_error = exc
                if attempt >= self.retries:
                    break
                time.sleep(0.5 * (attempt + 1))
        raise RuntimeError(str(last_error) if last_error else "Live request failed")

    def _fetch_players_payload(self) -> dict[str, Any]:
        rows = static_players.get_players()
        return {
            "players": [
                {
                    "player_id": int(row["id"]),
                    "player_name": row["full_name"],
                    "first_name": row["first_name"],
                    "last_name": row["last_name"],
                    "is_active": bool(row.get("is_active", False)),
                }
                for row in rows
            ]
        }

    def _fetch_rosters_payload(self) -> dict[str, Any]:
        teams_payload: dict[str, Any] = {}
        players_payload: dict[str, Any] = {}

        for team in self._teams:
            team_id = int(team["id"])
            roster_data = self._retry(
                lambda team_id=team_id: commonteamroster.CommonTeamRoster(
                    team_id=team_id,
                    season=self.season,
                    timeout=self.timeout,
                ).get_normalized_dict()
            )
            roster_rows = roster_data.get("CommonTeamRoster", [])
            player_rows: list[dict[str, Any]] = []

            for row in roster_rows:
                player = {
                    "player_id": int(row.get("PLAYER_ID") or row.get("PERSON_ID") or 0),
                    "player_name": row.get("PLAYER", "").strip(),
                    "position": row.get("POSITION", ""),
                    "jersey": row.get("NUM", ""),
                    "team_abbrev": team["abbreviation"],
                    "team_name": team["full_name"],
                    "team_id": team_id,
                    "season": self.season,
                }
                if not player["player_name"]:
                    continue
                player_rows.append(player)
                players_payload[normalize_lookup_key(player["player_name"])] = player

            teams_payload[team["abbreviation"]] = {
                "team_id": team_id,
                "team_abbrev": team["abbreviation"],
                "team_name": team["full_name"],
                "players": player_rows,
            }

        return {
            "season": self.season,
            "teams": teams_payload,
            "players": players_payload,
        }

    def _fetch_schedule_payload(self, game_date: str) -> dict[str, Any]:
        payload = self._retry(
            lambda: scoreboardv2.ScoreboardV2(
                game_date=self._format_game_date(game_date),
                day_offset=0,
                league_id="00",
                timeout=self.timeout,
            ).get_normalized_dict()
        )
        games: list[dict[str, Any]] = []
        for row in payload.get("GameHeader", []):
            home_team = self._team_by_id.get(int(row["HOME_TEAM_ID"]))
            away_team = self._team_by_id.get(int(row["VISITOR_TEAM_ID"]))
            if not home_team or not away_team:
                continue
            games.append({
                "game_id": row.get("GAME_ID"),
                "game_date": game_date,
                "home_team_abbrev": home_team["abbreviation"],
                "home_team_name": home_team["full_name"],
                "away_team_abbrev": away_team["abbreviation"],
                "away_team_name": away_team["full_name"],
                "status": row.get("GAME_STATUS_TEXT", ""),
            })
        return {"game_date": game_date, "games": games}

    @staticmethod
    def _format_game_date(game_date: str) -> str:
        year, month, day = game_date.split("-")
        return f"{month}/{day}/{year}"

    def refresh_players_cache(self, force: bool = False) -> dict[str, Any]:
        result = self.cache.get_or_refresh("players", self._fetch_players_payload, force_refresh=force)
        return {
            "payload": result.payload,
            "fetched_at": result.fetched_at,
            "stale": result.stale,
            "source": result.source,
            "warning": result.warning,
        }

    def refresh_rosters_cache(self, force: bool = False) -> dict[str, Any]:
        result = self.cache.get_or_refresh("rosters", self._fetch_rosters_payload, force_refresh=force)
        return {
            "payload": result.payload,
            "fetched_at": result.fetched_at,
            "stale": result.stale,
            "source": result.source,
            "warning": result.warning,
        }

    def refresh_schedule_cache(self, game_date: str, force: bool = False) -> dict[str, Any]:
        result = self.cache.get_or_refresh(
            f"schedule_{game_date}",
            lambda: self._fetch_schedule_payload(game_date),
            force_refresh=force,
        )
        return {
            "payload": result.payload,
            "fetched_at": result.fetched_at,
            "stale": result.stale,
            "source": result.source,
            "warning": result.warning,
        }

    def refresh_all(self, game_dates: list[str] | None = None, force: bool = False) -> dict[str, Any]:
        summary = {
            "players": self.refresh_players_cache(force=force),
            "rosters": self.refresh_rosters_cache(force=force),
            "schedules": {},
        }
        for game_date in game_dates or []:
            summary["schedules"][game_date] = self.refresh_schedule_cache(game_date, force=force)
        return summary

    def get_players(self) -> dict[str, Any]:
        return self.refresh_players_cache(force=False)["payload"]

    def get_rosters(self) -> dict[str, Any]:
        return self.refresh_rosters_cache(force=False)["payload"]

    def get_schedule(self, game_date: str) -> dict[str, Any]:
        return self.refresh_schedule_cache(game_date, force=False)["payload"]

    def get_team_roster(self, team_abbr_or_name: str) -> list[dict[str, Any]]:
        team_abbrev = resolve_team(team_abbr_or_name)
        if not team_abbrev:
            return []
        rosters = self.get_rosters()
        return rosters.get("teams", {}).get(team_abbrev, {}).get("players", [])

    def _player_candidates(self, player_name: str) -> list[dict[str, Any]]:
        rosters = self.get_rosters()
        live_players = list(rosters.get("players", {}).values())
        query = normalize_lookup_key(player_name)
        if not query:
            return []

        exact = [player for player in live_players if normalize_lookup_key(player["player_name"]) == query]
        if exact:
            return exact

        query_tokens = query.split()
        candidates: list[tuple[int, dict[str, Any]]] = []
        for player in live_players:
            full_name = normalize_lookup_key(player["player_name"])
            tokens = full_name.split()
            score = 0
            if query in full_name:
                score += 4
            if tokens and query == tokens[-1]:
                score += 3
            if any(token == query for token in tokens):
                score += 2
            if query_tokens and all(token in tokens for token in query_tokens):
                score += 5
            if score:
                candidates.append((score, player))

        candidates.sort(key=lambda item: (-item[0], item[1]["player_name"]))
        return [player for _, player in candidates]

    def resolve_player(self, player_name: str) -> dict[str, Any] | None:
        candidates = self._player_candidates(player_name)
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            best = candidates[0]
            second = candidates[1]
            if normalize_lookup_key(best["player_name"]) == normalize_lookup_key(player_name):
                return best
            if normalize_lookup_key(best["player_name"]).split()[-1] == normalize_lookup_key(player_name) and best["player_name"] != second["player_name"]:
                return best
            return None

        static_payload = self.get_players()
        query = normalize_lookup_key(player_name)
        fallback = []
        for player in static_payload.get("players", []):
            full_name = normalize_lookup_key(player["player_name"])
            if query == full_name or query in full_name or full_name.endswith(query):
                fallback.append(player)
        if len(fallback) == 1:
            return fallback[0]
        return None

    def get_current_team(self, player_name: str) -> dict[str, Any]:
        player = self.resolve_player(player_name)
        if not player:
            return {
                "found": False,
                "player_name": player_name,
                "verified_source": "nba_api",
            }

        team_abbrev = player.get("team_abbrev")
        team_name = player.get("team_name")
        return {
            "found": True,
            "player_id": player.get("player_id"),
            "player_name": player.get("player_name", player_name),
            "current_team_abbrev": team_abbrev,
            "current_team": team_name,
            "verified_source": "nba_api",
        }

    def is_player_on_team(self, player_name: str, team: str) -> bool:
        info = self.get_current_team(player_name)
        if not info.get("found"):
            return False
        team_abbrev = resolve_team(team)
        return bool(team_abbrev and team_abbrev == info.get("current_team_abbrev"))

    def validate_game(self, home_team: str, away_team: str, game_date: str | None) -> dict[str, Any]:
        home_abbrev = resolve_team(home_team)
        away_abbrev = resolve_team(away_team)
        if not home_abbrev or not away_abbrev:
            return {
                "ok": False,
                "game_exists": None,
                "reason": "team_resolution_failed",
            }
        if not game_date or not VERIFY_SCHEDULES:
            return {
                "ok": True,
                "game_exists": None,
                "home_team_abbrev": home_abbrev,
                "away_team_abbrev": away_abbrev,
                "verified_source": "nba_api",
                "schedule_checked": False,
            }

        try:
            schedule = self.get_schedule(game_date)
        except Exception as exc:
            log.warning("Live schedule validation failed for %s vs %s on %s: %s", home_abbrev, away_abbrev, game_date, exc)
            return {
                "ok": False,
                "game_exists": None,
                "reason": "schedule_unavailable",
                "warning": str(exc),
                "verified_source": "nba_api",
                "schedule_checked": True,
            }

        for game in schedule.get("games", []):
            teams = {game["home_team_abbrev"], game["away_team_abbrev"]}
            if teams == {home_abbrev, away_abbrev}:
                return {
                    "ok": True,
                    "game_exists": True,
                    "game_date": game_date,
                    "home_team_abbrev": game["home_team_abbrev"],
                    "home_team_name": game["home_team_name"],
                    "away_team_abbrev": game["away_team_abbrev"],
                    "away_team_name": game["away_team_name"],
                    "status": game.get("status", ""),
                    "verified_source": "nba_api",
                    "schedule_checked": True,
                }

        return {
            "ok": False,
            "game_exists": False,
            "reason": "game_not_found",
            "game_date": game_date,
            "home_team_abbrev": home_abbrev,
            "home_team_name": resolve_team_name(home_abbrev),
            "away_team_abbrev": away_abbrev,
            "away_team_name": resolve_team_name(away_abbrev),
            "verified_source": "nba_api",
            "schedule_checked": True,
        }

    def resolve_player_game_context(
        self,
        player_name: str,
        home_team: str,
        away_team: str,
        game_date: str | None = None,
    ) -> dict[str, Any]:
        player = self.get_current_team(player_name)
        if not player.get("found"):
            return {
                "ok": False,
                "reason": "player_not_found",
                "player_name": player_name,
            }

        home_abbrev = resolve_team(home_team)
        away_abbrev = resolve_team(away_team)
        live_team = player.get("current_team_abbrev")
        if live_team not in {home_abbrev, away_abbrev}:
            return {
                "ok": False,
                "reason": "player_not_in_game_context",
                "player_name": player["player_name"],
                "current_team": player.get("current_team"),
                "current_team_abbrev": live_team,
            }

        game_validation = self.validate_game(home_abbrev, away_abbrev, game_date)
        if game_date and not game_validation.get("ok"):
            return {
                "ok": False,
                "reason": game_validation.get("reason", "game_validation_failed"),
                "player_name": player["player_name"],
                "current_team": player.get("current_team"),
                "game_validation": game_validation,
            }

        actual_home = game_validation.get("home_team_abbrev", home_abbrev)
        is_home = 1 if live_team == actual_home else 0
        opponent = away_abbrev if is_home else home_abbrev
        return {
            "ok": True,
            "player_name": player["player_name"],
            "player_id": player.get("player_id"),
            "team_abbrev": live_team,
            "team_name": player.get("current_team"),
            "opponent_abbrev": opponent,
            "is_home": is_home,
            "game_validation": game_validation,
            "verified_source": "nba_api",
        }


@lru_cache(maxsize=1)
def get_live_client() -> NbaLiveClient:
    return NbaLiveClient()