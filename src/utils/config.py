"""Global configuration loaded from environment / .env file."""

import os
import re
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = Path(os.getenv("DATA_DIR", PROJECT_ROOT / "data"))
DB_PATH = Path(os.getenv("DB_PATH", PROJECT_ROOT / "data" / "processed" / "nba.duckdb"))
MODEL_DIR = Path(os.getenv("MODEL_DIR", PROJECT_ROOT / "models"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LIVE_CACHE_DIR = Path(os.getenv("LIVE_CACHE_DIR", DATA_DIR / "live"))
LIVE_CACHE_TTL_HOURS = int(os.getenv("LIVE_CACHE_TTL_HOURS", "12"))
LIVE_REQUEST_TIMEOUT = int(os.getenv("LIVE_REQUEST_TIMEOUT", "15"))
LIVE_API_RETRIES = int(os.getenv("LIVE_API_RETRIES", "2"))
ENABLE_LIVE_VERIFICATION = os.getenv("ENABLE_LIVE_VERIFICATION", "true").strip().lower() in {
    "1", "true", "yes", "on",
}
VERIFY_SCHEDULES = os.getenv("VERIFY_SCHEDULES", "true").strip().lower() in {
    "1", "true", "yes", "on",
}

# Ensure directories exist
MODEL_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH.parent.mkdir(parents=True, exist_ok=True)
LIVE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

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

TEAM_ALIASES = {
    "atlanta": "ATL",
    "hawks": "ATL",
    "boston": "BOS",
    "celtics": "BOS",
    "brooklyn": "BKN",
    "nets": "BKN",
    "charlotte": "CHA",
    "hornets": "CHA",
    "chicago": "CHI",
    "bulls": "CHI",
    "cleveland": "CLE",
    "cavs": "CLE",
    "cavaliers": "CLE",
    "dallas": "DAL",
    "mavericks": "DAL",
    "mavs": "DAL",
    "denver": "DEN",
    "nuggets": "DEN",
    "detroit": "DET",
    "pistons": "DET",
    "golden state": "GSW",
    "golden state warriors": "GSW",
    "warriors": "GSW",
    "houston": "HOU",
    "rockets": "HOU",
    "indiana": "IND",
    "pacers": "IND",
    "clippers": "LAC",
    "la clippers": "LAC",
    "los angeles clippers": "LAC",
    "lake show": "LAL",
    "lakers": "LAL",
    "la lakers": "LAL",
    "los angeles lakers": "LAL",
    "memphis": "MEM",
    "grizzlies": "MEM",
    "miami": "MIA",
    "heat": "MIA",
    "milwaukee": "MIL",
    "bucks": "MIL",
    "minnesota": "MIN",
    "timberwolves": "MIN",
    "wolves": "MIN",
    "new orleans": "NOP",
    "pelicans": "NOP",
    "new york": "NYK",
    "knicks": "NYK",
    "okc": "OKC",
    "oklahoma city": "OKC",
    "thunder": "OKC",
    "orlando": "ORL",
    "magic": "ORL",
    "philadelphia": "PHI",
    "sixers": "PHI",
    "76ers": "PHI",
    "philly": "PHI",
    "phoenix": "PHX",
    "suns": "PHX",
    "portland": "POR",
    "trail blazers": "POR",
    "blazers": "POR",
    "sacramento": "SAC",
    "kings": "SAC",
    "san antonio": "SAS",
    "spurs": "SAS",
    "toronto": "TOR",
    "raptors": "TOR",
    "utah": "UTA",
    "jazz": "UTA",
    "washington": "WAS",
    "wizards": "WAS",
}


def normalize_lookup_key(value: str) -> str:
    """Normalize user input for flexible team/player matching."""
    return re.sub(r"[^a-z0-9]+", " ", value.strip().lower()).strip()


def current_nba_season(today: date | None = None) -> str:
    """Return the current NBA season in `YYYY-YY` format."""
    today = today or date.today()
    start_year = today.year if today.month >= 10 else today.year - 1
    return f"{start_year}-{str(start_year + 1)[-2:]}"


LIVE_SEASON = os.getenv("LIVE_SEASON", current_nba_season())


def resolve_team_name(name: str) -> str | None:
    """Resolve a flexible team name string to the canonical full team name."""
    abbrev = resolve_team(name)
    if not abbrev:
        return None
    return TEAM_ABBREV_MAP[abbrev]

def resolve_team(name: str) -> str | None:
    """Resolve a flexible team name string to a 3-letter abbreviation."""
    if not name:
        return None
    name = name.strip()
    up = name.upper()
    if up in TEAM_ABBREV_MAP:
        return up
    if name in FULL_NAME_TO_ABBREV:
        return FULL_NAME_TO_ABBREV[name]
    norm = normalize_lookup_key(name)
    if norm in TEAM_ALIASES:
        return TEAM_ALIASES[norm]
    # Try partial match
    low = norm
    for full, abbrev in FULL_NAME_TO_ABBREV.items():
        full_norm = normalize_lookup_key(full)
        if low == full_norm or low in full_norm:
            return abbrev
    for abbrev in TEAM_ABBREV_MAP:
        if low == abbrev.lower():
            return abbrev
    return None
