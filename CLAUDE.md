# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NBA Parlay Maker is a local **Model Context Protocol (MCP) server** that provides NBA betting analytics. It uses XGBoost models trained on historical box-score data to predict player props and moneylines, with a live verification layer that validates picks against current NBA rosters and schedules.

## Commands

### Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

### Data & Training Pipeline
```bash
# Load CSVs into DuckDB (idempotent; use --force to rebuild)
python scripts/ingest.py [--force]

# Train all 5 models (player_points, player_rebounds, player_assists, player_pra, moneyline)
python scripts/train.py

# Train a single model
python scripts/train.py player_points

# Refresh live roster/schedule cache
python scripts/update_live_cache.py
```

### Running the Server
```bash
python scripts/run_server.py
```
Listens on stdin/stdout using MCP JSON-RPC 2.0. Integrates with Claude Desktop via `claude_desktop_config.json`.

### Tests
```bash
python -m pytest tests/test_live_verification.py -v

# Run a single test
python -m pytest tests/test_live_verification.py::TestClassName::test_method -v
```

There is no configured linter or formatter (no `.flake8`, `pyproject.toml`, or `setup.cfg`).

## Architecture

### Data Flow
```
CSV files (data/)
  → scripts/ingest.py → DuckDB (data/processed/nba.duckdb)
  → src/features/engineering.py (rolling features, matchup stats)
  → src/models/train.py → XGBoost models (models/*.joblib)
  → src/models/predict.py → predictions + confidence scores
  → src/tools/betting_tools.py → MCP tool handlers
  → src/mcp_server/server.py → FastMCP server (Claude Desktop)
```

### Key Components

**`src/models/predict.py`** — Inference layer. `predict_player_prop()` builds a feature vector (rolling averages over 3/5/10 game windows, consistency, matchup history) and returns a projected stat + hit probability. `predict_moneyline()` returns home win probability. `confidence_score()` combines hit probability (50%) + data quality (25%) + consistency (25%); tiers are High ≥ 0.65, Medium ≥ 0.45, Low < 0.45.

**`src/features/engineering.py`** — Feature construction. All features are computed from games strictly before the target date to prevent leakage. Includes exponential-decay weighted recent form, home/away splits, and opponent matchup history.

**`src/parlay/engine.py`** — Parlay assembly. Correlation penalty formula: +0.04 per same-game pair, +0.06 per same-team pair, +0.10 for same player, capped at 0.40 (halved if `allow_correlation=True`). Risk modes: `safe` (max 2 legs, high confidence only), `balanced` (max 3 legs), `aggressive` (up to 6 legs).

**`src/verification/`** — Live verification layer. `nba_live.py` wraps `nba_api` with TTL caching and retry logic. `verify_pick.py` validates each parlay leg: game must exist on the date, player must be on the correct team, player must be active. Falls back to stale cache if the API is unavailable.

**`src/tools/betting_tools.py`** — Implements the 7 MCP tools: `analyze_game`, `evaluate_moneyline`, `evaluate_player_prop`, `suggest_player_props_tool`, `make_parlay_tool`, `explain_leg`, `find_best_legs_tool`, `fade_risky_legs_tool`. Each tool parses dates (defaults to today), resolves team/player names, runs live verification, calls prediction models, and returns JSON with a disclaimer.

**`src/utils/config.py`** — Team name normalization maps 100+ aliases to abbreviations (e.g., "Lakers" / "lake show" / "los angeles lakers" → "LAL"). Also handles season string formatting and loads `.env`.

**`src/data/queries.py`** — SQL helpers over DuckDB: `player_recent()`, `player_vs_opponent()`, `team_recent()`, `team_vs_team()`.

### DuckDB Schema
| Table | Key Columns |
|-------|-------------|
| `games` | gameId, gameDateTimeEst, homeScore, awayScore, home_abbrev, away_abbrev |
| `players` | personId, firstName, lastName |
| `player_box` | personId, gameId, game_date, points, assists, reboundsTotal, numMinutes, team_abbrev, opp_abbrev, home, pra |
| `team_box` | gameId, teamId, teamScore, opponentScore, home, win |
| `league_schedule` | gameId, gameDateTimeEst, homeTeamName, awayTeamName |

### Environment Variables (`.env`)
| Variable | Default | Purpose |
|----------|---------|---------|
| `DATA_DIR` | `./data` | CSV input directory |
| `DB_PATH` | `./data/processed/nba.duckdb` | DuckDB path |
| `MODEL_DIR` | `./models` | Trained model directory |
| `LIVE_CACHE_DIR` | `./data/live` | Live roster/schedule cache |
| `LIVE_CACHE_TTL_HOURS` | `12` | Cache validity period |
| `ENABLE_LIVE_VERIFICATION` | `true` | Toggle live roster checks |
| `VERIFY_SCHEDULES` | `true` | Toggle schedule verification |

### Important Notes
- `data/` and `models/` are git-ignored; they must be generated locally via `ingest.py` and `train.py`
- Models use a time-aware train/validation split (60-day validation window) to prevent leakage
- The MCP server uses FastMCP over stdio transport; there is no HTTP interface
- Live verification was added recently (commit `08c6693`) — it is the primary guard against stale/invalid picks
