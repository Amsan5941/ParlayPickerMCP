"""
Data ingestion pipeline.
Loads CSV files from the data directory, validates schemas,
normalises column names, and loads into a local DuckDB database.
"""

import duckdb
import pandas as pd
from pathlib import Path

from src.utils.config import DATA_DIR, DB_PATH, CITY_NAME_TO_ABBREV
from src.utils.logger import get_logger

log = get_logger(__name__)

# Expected CSV files and their required columns (subset for validation)
EXPECTED_FILES = {
    "Games.csv": ["gameId", "gameDateTimeEst", "hometeamName", "awayteamName",
                   "homeScore", "awayScore", "winner"],
    "Players.csv": ["personId", "firstName", "lastName"],
    "PlayerStatistics.csv": ["personId", "gameId", "gameDateTimeEst", "points",
                              "assists", "reboundsTotal", "numMinutes"],
    "TeamStatistics.csv": ["gameId", "teamId", "teamName", "teamScore",
                            "opponentScore", "home", "win"],
    "LeagueSchedule25_26.csv": ["gameId", "gameDateTimeEst", "homeTeamName", "awayTeamName"],
}


def _add_team_abbrev(df: pd.DataFrame, city_col: str, name_col: str, out_col: str) -> pd.DataFrame:
    """Add a team abbreviation column from city+name columns."""
    def _lookup(row):
        key = (str(row[city_col]).strip(), str(row[name_col]).strip())
        return CITY_NAME_TO_ABBREV.get(key, "")
    df[out_col] = df.apply(_lookup, axis=1)
    return df


def load_csv(filename: str) -> pd.DataFrame:
    """Load a CSV from DATA_DIR with basic validation."""
    path = DATA_DIR / filename
    if not path.exists():
        log.warning("File not found: %s", path)
        return pd.DataFrame()
    log.info("Loading %s ...", filename)
    df = pd.read_csv(path, low_memory=False)
    required = EXPECTED_FILES.get(filename, [])
    missing = [c for c in required if c not in df.columns]
    if missing:
        log.warning("Missing columns in %s: %s", filename, missing)
    return df


def ingest_all(force: bool = False) -> duckdb.DuckDBPyConnection:
    """Load all CSVs into DuckDB. Returns the connection."""
    if DB_PATH.exists() and not force:
        log.info("Database already exists at %s. Use force=True to rebuild.", DB_PATH)
        return duckdb.connect(str(DB_PATH))

    log.info("Building DuckDB database at %s", DB_PATH)
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DB_PATH))

    # --- Games ---
    games = load_csv("Games.csv")
    if not games.empty:
        games["gameDateTimeEst"] = pd.to_datetime(games["gameDateTimeEst"], errors="coerce")
        games["game_date"] = games["gameDateTimeEst"].dt.date
        games = _add_team_abbrev(games, "hometeamCity", "hometeamName", "home_abbrev")
        games = _add_team_abbrev(games, "awayteamCity", "awayteamName", "away_abbrev")
        # Filter to regular season only for modelling
        games_reg = games[games["gameType"] == "Regular Season"].copy()
        con.execute("DROP TABLE IF EXISTS games")
        con.execute("CREATE TABLE games AS SELECT * FROM games_reg")
        log.info("  games table: %d rows", len(games_reg))

    # --- Players ---
    players = load_csv("Players.csv")
    if not players.empty:
        con.execute("DROP TABLE IF EXISTS players")
        con.execute("CREATE TABLE players AS SELECT * FROM players")
        log.info("  players table: %d rows", len(players))

    # --- Player Statistics (box scores) ---
    ps = load_csv("PlayerStatistics.csv")
    if not ps.empty:
        ps["gameDateTimeEst"] = pd.to_datetime(ps["gameDateTimeEst"], errors="coerce")
        ps["game_date"] = ps["gameDateTimeEst"].dt.date
        # Keep regular season only
        ps = ps[ps["gameType"] == "Regular Season"].copy()
        ps = _add_team_abbrev(ps, "playerteamCity", "playerteamName", "team_abbrev")
        ps = _add_team_abbrev(ps, "opponentteamCity", "opponentteamName", "opp_abbrev")
        # Compute PRA
        for col in ["points", "assists", "reboundsTotal"]:
            ps[col] = pd.to_numeric(ps[col], errors="coerce").fillna(0)
        ps["pra"] = ps["points"] + ps["reboundsTotal"] + ps["assists"]
        ps["numMinutes"] = pd.to_numeric(ps["numMinutes"], errors="coerce").fillna(0)
        ps["player_name"] = ps["firstName"].str.strip() + " " + ps["lastName"].str.strip()
        con.execute("DROP TABLE IF EXISTS player_box")
        con.execute("CREATE TABLE player_box AS SELECT * FROM ps")
        log.info("  player_box table: %d rows", len(ps))

    # --- Team Statistics ---
    ts = load_csv("TeamStatistics.csv")
    if not ts.empty:
        ts["gameDateTimeEst"] = pd.to_datetime(ts["gameDateTimeEst"], errors="coerce")
        ts["game_date"] = ts["gameDateTimeEst"].dt.date
        ts = ts[ts.get("gameType", ts.columns[0]) != ""].copy() if "gameType" in ts.columns else ts
        ts = _add_team_abbrev(ts, "teamCity", "teamName", "team_abbrev")
        ts = _add_team_abbrev(ts, "opponentTeamCity", "opponentTeamName", "opp_abbrev")
        con.execute("DROP TABLE IF EXISTS team_box")
        con.execute("CREATE TABLE team_box AS SELECT * FROM ts")
        log.info("  team_box table: %d rows", len(ts))

    # --- Team Statistics Advanced ---
    tsa = load_csv("TeamStatisticsAdvanced.csv") if (DATA_DIR / "TeamStatisticsAdvanced.csv").exists() else pd.DataFrame()
    if not tsa.empty:
        tsa["gameDateTimeEst"] = pd.to_datetime(tsa["gameDateTimeEst"], errors="coerce")
        tsa["game_date"] = tsa["gameDateTimeEst"].dt.date
        con.execute("DROP TABLE IF EXISTS team_advanced")
        con.execute("CREATE TABLE team_advanced AS SELECT * FROM tsa")
        log.info("  team_advanced table: %d rows", len(tsa))

    # --- Schedule ---
    for sched_file, tbl_name in [("LeagueSchedule24_25.csv", "schedule_24_25"),
                                  ("LeagueSchedule25_26.csv", "schedule_25_26")]:
        sdf = load_csv(sched_file) if (DATA_DIR / sched_file).exists() else pd.DataFrame()
        if not sdf.empty:
            sdf["gameDateTimeEst"] = pd.to_datetime(sdf["gameDateTimeEst"], errors="coerce")
            sdf["game_date"] = sdf["gameDateTimeEst"].dt.date
            con.execute(f"DROP TABLE IF EXISTS {tbl_name}")
            con.execute(f"CREATE TABLE {tbl_name} AS SELECT * FROM sdf")
            log.info("  %s table: %d rows", tbl_name, len(sdf))

    # --- Player Stats Advanced ---
    psa = load_csv("PlayerStatisticsAdvanced.csv") if (DATA_DIR / "PlayerStatisticsAdvanced.csv").exists() else pd.DataFrame()
    if not psa.empty:
        psa["gameDateTimeEst"] = pd.to_datetime(psa["gameDateTimeEst"], errors="coerce")
        psa["game_date"] = psa["gameDateTimeEst"].dt.date
        con.execute("DROP TABLE IF EXISTS player_advanced")
        con.execute("CREATE TABLE player_advanced AS SELECT * FROM psa")
        log.info("  player_advanced table: %d rows", len(psa))

    log.info("Ingestion complete.")
    return con


def get_connection() -> duckdb.DuckDBPyConnection:
    """Get a connection to the existing DB, or ingest first if needed."""
    if not DB_PATH.exists():
        return ingest_all()
    return duckdb.connect(str(DB_PATH))
