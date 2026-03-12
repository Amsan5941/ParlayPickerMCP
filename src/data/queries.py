"""
Convenience query helpers on top of DuckDB.
"""

import pandas as pd
import duckdb
from datetime import date, timedelta

from src.data.ingest import get_connection
from src.utils.config import resolve_team
from src.utils.logger import get_logger

log = get_logger(__name__)


def _con() -> duckdb.DuckDBPyConnection:
    return get_connection()


def player_game_log(player_name: str, seasons_back: int = 3,
                    game_type: str = "Regular Season") -> pd.DataFrame:
    """Return a player's box-score game log ordered by date descending."""
    con = _con()
    q = """
        SELECT * FROM player_box
        WHERE player_name ILIKE ?
          AND game_date >= CURRENT_DATE - INTERVAL ? YEAR
        ORDER BY game_date DESC
    """
    return con.execute(q, [f"%{player_name}%", seasons_back]).fetchdf()


def player_recent(player_name: str, n: int = 10) -> pd.DataFrame:
    """Last n games for a player."""
    con = _con()
    q = """
        SELECT * FROM player_box
        WHERE player_name ILIKE ?
        ORDER BY game_date DESC
        LIMIT ?
    """
    return con.execute(q, [f"%{player_name}%", n]).fetchdf()


def player_vs_opponent(player_name: str, opp_abbrev: str, n: int = 20) -> pd.DataFrame:
    """Player's games against a specific opponent."""
    con = _con()
    q = """
        SELECT * FROM player_box
        WHERE player_name ILIKE ?
          AND opp_abbrev = ?
        ORDER BY game_date DESC
        LIMIT ?
    """
    return con.execute(q, [f"%{player_name}%", opp_abbrev, n]).fetchdf()


def team_recent(team_abbrev: str, n: int = 10) -> pd.DataFrame:
    """Last n team box scores."""
    con = _con()
    q = """
        SELECT * FROM team_box
        WHERE team_abbrev = ?
        ORDER BY game_date DESC
        LIMIT ?
    """
    return con.execute(q, [team_abbrev, n]).fetchdf()


def team_vs_team(team1: str, team2: str, n: int = 20) -> pd.DataFrame:
    """Head-to-head games between two teams (from team1's perspective)."""
    con = _con()
    q = """
        SELECT * FROM team_box
        WHERE team_abbrev = ? AND opp_abbrev = ?
        ORDER BY game_date DESC
        LIMIT ?
    """
    return con.execute(q, [team1, team2, n]).fetchdf()


def games_on_date(game_date: str) -> pd.DataFrame:
    """All games on a specific date."""
    con = _con()
    q = """
        SELECT * FROM games
        WHERE game_date = CAST(? AS DATE)
    """
    return con.execute(q, [game_date]).fetchdf()


def schedule_on_date(game_date: str) -> pd.DataFrame:
    """Check schedule tables for games on a date."""
    con = _con()
    for tbl in ["schedule_25_26", "schedule_24_25"]:
        try:
            q = f"SELECT * FROM {tbl} WHERE game_date = CAST(? AS DATE)"
            df = con.execute(q, [game_date]).fetchdf()
            if not df.empty:
                return df
        except Exception:
            continue
    return pd.DataFrame()


def team_season_stats(team_abbrev: str, last_n: int = 20) -> pd.DataFrame:
    """Aggregated team stats over last N games."""
    con = _con()
    q = """
        SELECT
            AVG(teamScore) as avg_score,
            AVG(opponentScore) as avg_opp_score,
            AVG(teamScore) - AVG(opponentScore) as avg_margin,
            SUM(CASE WHEN win=1 THEN 1 ELSE 0 END) as wins,
            COUNT(*) as games,
            AVG(reboundsTotal) as avg_reb,
            AVG(assists) as avg_ast,
            AVG(turnovers) as avg_tov,
            STDDEV(teamScore) as score_stddev
        FROM team_box
        WHERE team_abbrev = ?
        ORDER BY game_date DESC
        LIMIT ?
    """
    return con.execute(q, [team_abbrev, last_n]).fetchdf()


def find_player_name(partial: str) -> str | None:
    """Resolve a partial player name to exact name in DB."""
    con = _con()
    q = """
        SELECT DISTINCT player_name FROM player_box
        WHERE player_name ILIKE ?
        ORDER BY game_date DESC
        LIMIT 1
    """
    df = con.execute(q, [f"%{partial}%"]).fetchdf()
    if df.empty:
        return None
    return df.iloc[0]["player_name"]


def player_last_game_date(player_name: str) -> date | None:
    """Date of a player's most recent game."""
    con = _con()
    q = """
        SELECT MAX(game_date) as last_date FROM player_box
        WHERE player_name ILIKE ?
    """
    df = con.execute(q, [f"%{player_name}%"]).fetchdf()
    if df.empty or df.iloc[0]["last_date"] is None:
        return None
    return pd.Timestamp(df.iloc[0]["last_date"]).date()


def team_last_game_date(team_abbrev: str) -> date | None:
    """Date of a team's most recent game."""
    con = _con()
    q = """
        SELECT MAX(game_date) as last_date FROM team_box
        WHERE team_abbrev = ?
    """
    df = con.execute(q, [team_abbrev]).fetchdf()
    if df.empty or df.iloc[0]["last_date"] is None:
        return None
    return pd.Timestamp(df.iloc[0]["last_date"]).date()
