"""Global configuration loaded from environment / .env file."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = Path(os.getenv("DATA_DIR", PROJECT_ROOT / "data"))
DB_PATH = Path(os.getenv("DB_PATH", PROJECT_ROOT / "data" / "processed" / "nba.duckdb"))
MODEL_DIR = Path(os.getenv("MODEL_DIR", PROJECT_ROOT / "models"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Ensure directories exist
MODEL_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# Team name normalisation helpers
TEAM_ABBREV_MAP = {
    "ATL": "Atlanta Hawks", "BOS": "Boston Celtics", "BKN": "Brooklyn Nets",
    "CHA": "Charlotte Hornets", "CHI": "Chicago Bulls", "CLE": "Cleveland Cavaliers",
    "DAL": "Dallas Mavericks", "DEN": "Denver Nuggets", "DET": "Detroit Pistons",
    "GSW": "Golden State Warriors", "HOU": "Houston Rockets", "IND": "Indiana Pacers",
    "LAC": "LA Clippers", "LAL": "Los Angeles Lakers", "MEM": "Memphis Grizzlies",
    "MIA": "Miami Heat", "MIL": "Milwaukee Bucks", "MIN": "Minnesota Timberwolves",
    "NOP": "New Orleans Pelicans", "NYK": "New York Knicks", "OKC": "Oklahoma City Thunder",
    "ORL": "Orlando Magic", "PHI": "Philadelphia 76ers", "PHX": "Phoenix Suns",
    "POR": "Portland Trail Blazers", "SAC": "Sacramento Kings", "SAS": "San Antonio Spurs",
    "TOR": "Toronto Raptors", "UTA": "Utah Jazz", "WAS": "Washington Wizards",
}

FULL_NAME_TO_ABBREV = {v: k for k, v in TEAM_ABBREV_MAP.items()}

# Also build lookup by (city, name) tuple
CITY_NAME_TO_ABBREV: dict[tuple[str, str], str] = {}
for abbrev, full in TEAM_ABBREV_MAP.items():
    parts = full.rsplit(" ", 1)
    if len(parts) == 2:
        # Most teams: "City TeamName"
        CITY_NAME_TO_ABBREV[(parts[0], parts[1])] = abbrev
# Special multi-word names
CITY_NAME_TO_ABBREV[("Golden State", "Warriors")] = "GSW"
CITY_NAME_TO_ABBREV[("Oklahoma City", "Thunder")] = "OKC"
CITY_NAME_TO_ABBREV[("New Orleans", "Pelicans")] = "NOP"
CITY_NAME_TO_ABBREV[("New York", "Knicks")] = "NYK"
CITY_NAME_TO_ABBREV[("San Antonio", "Spurs")] = "SAS"
CITY_NAME_TO_ABBREV[("Los Angeles", "Lakers")] = "LAL"
CITY_NAME_TO_ABBREV[("Los Angeles", "Clippers")] = "LAC"
CITY_NAME_TO_ABBREV[("LA", "Clippers")] = "LAC"
CITY_NAME_TO_ABBREV[("Portland", "Trail Blazers")] = "POR"
CITY_NAME_TO_ABBREV[("Trail", "Blazers")] = "POR"

def resolve_team(name: str) -> str | None:
    """Resolve a flexible team name string to a 3-letter abbreviation."""
    name = name.strip()
    up = name.upper()
    if up in TEAM_ABBREV_MAP:
        return up
    if name in FULL_NAME_TO_ABBREV:
        return FULL_NAME_TO_ABBREV[name]
    # Try partial match
    low = name.lower()
    for full, abbrev in FULL_NAME_TO_ABBREV.items():
        if low in full.lower():
            return abbrev
    for abbrev in TEAM_ABBREV_MAP:
        if low == abbrev.lower():
            return abbrev
    return None
