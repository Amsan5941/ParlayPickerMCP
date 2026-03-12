# NBA Parlay Maker — MCP Server

A **local** Model Context Protocol (MCP) server for NBA betting analytics. Provides data-driven matchup analysis, player prop evaluation, moneyline predictions, and parlay generation powered by XGBoost models trained on historical NBA box-score data.

> **DISCLAIMER:** This is a probabilistic analytics tool, NOT financial advice. Sports betting carries inherent risk. Never bet more than you can afford to lose. Past performance does not predict future results. No bet is guaranteed to win.

---

## Project Structure

```
parlay_maker/
├── data/                       # Raw CSV files (place Kaggle data here)
│   ├── processed/              # DuckDB database (auto-generated)
│   ├── Games.csv
│   ├── Players.csv
│   ├── PlayerStatistics.csv
│   ├── PlayerStatisticsAdvanced.csv
│   ├── PlayerStatisticsMisc.csv
│   ├── PlayerStatisticsScoring.csv
│   ├── PlayerStatisticsUsage.csv
│   ├── TeamStatistics.csv
│   ├── TeamStatisticsAdvanced.csv
│   ├── TeamStatisticsFourFactors.csv
│   ├── TeamStatisticsMisc.csv
│   ├── TeamStatisticsScoring.csv
│   ├── TeamHistories.csv
│   ├── LeagueSchedule24_25.csv
│   └── LeagueSchedule25_26.csv
├── models/                     # Trained model files (auto-generated)
├── scripts/
│   ├── ingest.py               # Data ingestion CLI
│   ├── train.py                # Model training CLI
│   └── run_server.py           # MCP server runner
├── src/
│   ├── data/
│   │   ├── ingest.py           # CSV → DuckDB pipeline
│   │   └── queries.py          # Query helpers
│   ├── features/
│   │   └── engineering.py      # Feature engineering
│   ├── models/
│   │   ├── train.py            # Model training
│   │   └── predict.py          # Prediction interface
│   ├── parlay/
│   │   └── engine.py           # Parlay builder & scoring
│   ├── tools/
│   │   └── betting_tools.py    # MCP tool implementations
│   ├── mcp_server/
│   │   └── server.py           # MCP server (FastMCP)
│   └── utils/
│       ├── config.py           # Config & constants
│       └── logger.py           # Logging
├── tests/
├── notebooks/
├── requirements.txt
├── .env
├── .env.example
└── README.md
```

---

## Quick Start

### 1. Prerequisites

- Python 3.11+
- The Kaggle dataset CSVs (already in `data/`)

### 2. Install Dependencies

```bash
cd /Users/amsan/Downloads/parlay_maker
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Ingest Data

Loads all CSV files into a local DuckDB database:

```bash
python scripts/ingest.py
```

To rebuild from scratch:
```bash
python scripts/ingest.py --force
```

### 4. Train Models

Trains XGBoost models for moneyline, points, rebounds, assists, and PRA:

```bash
python scripts/train.py
```

To train a single model:
```bash
python scripts/train.py player_points
python scripts/train.py moneyline
```

Available models: `player_points`, `player_rebounds`, `player_assists`, `player_pra`, `moneyline`

### 5. Run the MCP Server

```bash
python scripts/run_server.py
```

### 6. Connect to Claude Desktop

Add this to your Claude Desktop config file:

**macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "nba-parlay-maker": {
      "command": "/Users/amsan/Downloads/parlay_maker/.venv/bin/python",
      "args": ["/Users/amsan/Downloads/parlay_maker/scripts/run_server.py"],
      "env": {
        "DATA_DIR": "/Users/amsan/Downloads/parlay_maker/data",
        "DB_PATH": "/Users/amsan/Downloads/parlay_maker/data/processed/nba.duckdb",
        "MODEL_DIR": "/Users/amsan/Downloads/parlay_maker/models"
      }
    }
  }
}
```

Restart Claude Desktop after saving.

---

## MCP Tools

### `analyze_game_tool`
Analyze a game matchup — projected winner, confidence, reasons, caution flags.

### `evaluate_moneyline_tool`
Evaluate a moneyline bet — win probabilities, lean, explanation.

### `evaluate_player_prop_tool`
Evaluate a player prop — projection, hit probability, confidence tier, risks.

### `suggest_props_tool`
Suggest the best props for a player — ranked by confidence across all stat types.

### `make_parlay_builder`
Build a parlay — ranked options with per-leg confidence, correlation warnings, combined probability.

### `explain_leg_tool`
Deep-dive into why the model likes/dislikes a specific leg.

### `find_best_legs_on_slate`
Find the strongest legs on a slate/game, ranked by confidence.

### `fade_risky_legs_on_slate`
Find legs to AVOID with risk explanations.

---

## 10 Example Prompts for Claude Desktop

Paste these directly into Claude Desktop:

1. **"Analyze the Lakers vs Celtics game today"**
2. **"Who wins ML in Nuggets vs Suns?"**
3. **"Evaluate LeBron James over 27.5 points vs Celtics"**
4. **"What's the best PRA play for Jokic against the Lakers?"**
5. **"Make me a safe 2-leg parlay for Knicks vs Heat with Jalen Brunson, Bam Adebayo, and Tyler Herro"**
6. **"Build an aggressive 4-leg same game parlay for Warriors vs Suns with Curry, Booker, Durant, and Wiggins"**
7. **"What's the safest leg on the board for Celtics vs 76ers with Tatum, Brown, Embiid, and Maxey?"**
8. **"Compare Tatum vs Brunson — who has the better PRA play tonight?"**
9. **"Which legs should I avoid in the Nuggets vs Suns game with Jokic, Murray, Booker, and Durant?"**
10. **"Give me 3 strong legs for tonight's Lakers game but avoid correlated picks — use LeBron, AD, and the opposing team's top players"**

---

## How It Works (Non-Technical)

1. **Data:** The system loads years of NBA game data — every player's box score (points, rebounds, assists, minutes, etc.) and team results.

2. **Features:** For each prediction, it calculates dozens of "signals" like:
   - How has this player performed in their last 3/5/10 games?
   - Are they trending up or down?
   - How do they do at home vs away?
   - How do they perform against THIS specific opponent?
   - Are they on a back-to-back? Rested?
   - How consistent are they? (Do they hit 25 points every night, or swing between 15 and 40?)

3. **Models:** Machine learning models (XGBoost) are trained on historical data to predict:
   - How many points/rebounds/assists a player will get
   - Which team will win

4. **Probability:** For any betting line (e.g., "LeBron over 27.5 points"), the system estimates the probability of it hitting by combining the model prediction with the player's historical distribution.

5. **Parlays:** The parlay engine picks the best legs, checks for dangerous correlations (e.g., two bets that fail together), and scores each combo by estimated hit probability.

6. **Safety:** It explicitly warns about risky legs, low confidence, small samples, and volatile players. It never claims a bet will "definitely hit."

---

## Confidence Scoring Formula

```
score = hit_probability × 0.50
      + data_quality     × 0.25    (min(sample_size/20, 1.0))
      + consistency      × 0.25    (1 - coefficient_of_variation)

Tiers:
  High   ≥ 0.65
  Medium ≥ 0.45
  Low    < 0.45
```

## Correlation Penalty Formula

```
penalty = 0.0
For each pair of legs:
  + 0.04  if same game
  + 0.06  if same team within same game
  + 0.10  if same player (guard)
Capped at 0.40 (40% max penalty)

adjusted_probability = raw_combined × (1 - penalty)
```

---

## Example Outputs

### Moneyline Request
```json
{
  "game": "LAL vs BOS",
  "home_win_probability": 0.42,
  "away_win_probability": 0.58,
  "lean": "BOS (Away)",
  "confidence_tier": "Medium",
  "reasons": [
    "BOS last-10 win rate: 70%",
    "LAL avg margin (10g): +2.3",
    "BOS avg margin (10g): +6.1"
  ],
  "caution_flags": []
}
```

### Player PRA Request
```json
{
  "player": "Nikola Jokic",
  "stat_type": "pra",
  "line": 46.5,
  "over_under": "over",
  "projection": 51.3,
  "hit_probability": 0.72,
  "confidence_tier": "High",
  "reasons": [
    "Model projects 51.3, which is 4.8 above the 46.5 line",
    "Trending UP: last 5 avg 53.2 vs 10-game avg 49.8",
    "Averages 48.7 in 6 games vs LAL"
  ],
  "risks": [],
  "suggested_usage": "parlay-safe"
}
```

### 2-Leg Safe Parlay
```json
{
  "name": "Safe 2-Leg Parlay",
  "mode": "safe",
  "legs": [
    {
      "description": "Nikola Jokic Over 46.5 pra",
      "hit_probability": 0.72,
      "confidence_tier": "High"
    },
    {
      "description": "DEN ML",
      "hit_probability": 0.62,
      "confidence_tier": "Medium"
    }
  ],
  "combined_probability": 0.4464,
  "correlation_penalty": 0.04,
  "adjusted_probability": 0.4285,
  "reasoning": "Selected high-confidence, high-consistency legs. Same-game parlay — correlated outcomes possible.",
  "warnings": ["Correlation penalty of 4% applied — legs may be dependent"]
}
```

### 4-Leg Aggressive SGP
```json
{
  "name": "Aggressive 4-Leg Parlay",
  "mode": "aggressive",
  "legs": [
    {
      "description": "Jokic Over 48.5 pra",
      "hit_probability": 0.65,
      "confidence_tier": "High"
    },
    {
      "description": "Murray Over 22.5 points",
      "hit_probability": 0.58,
      "confidence_tier": "Medium"
    },
    {
      "description": "Booker Over 27.5 points",
      "hit_probability": 0.55,
      "confidence_tier": "Medium"
    },
    {
      "description": "Durant Over 7.5 rebounds",
      "hit_probability": 0.52,
      "confidence_tier": "Medium"
    }
  ],
  "combined_probability": 0.1078,
  "correlation_penalty": 0.14,
  "adjusted_probability": 0.0927,
  "reasoning": "Selected higher-upside legs with more variance. Same-game parlay — correlated outcomes possible.",
  "warnings": [
    "Correlation penalty of 14% applied — legs may be dependent",
    "4-leg parlays have inherently low hit rates — use small stake only"
  ]
}
```

---

## Retraining Models

When new data is available:

```bash
# 1. Place updated CSV files in data/
# 2. Re-ingest
python scripts/ingest.py --force
# 3. Retrain
python scripts/train.py
```

---

## V2 Improvement Ideas

1. **Live Odds API Integration:** Connect to a sportsbook API (e.g., The Odds API) to automatically pull current lines instead of requiring manual input. Compare model projections against market lines to find edge/value.

2. **Injury/Lineup Integration:** Add a real-time injury feed (e.g., from NBA API or RotoWire) to auto-detect missing players, adjust projections for players inheriting additional minutes/usage.

3. **Position-Based Opponent Defense:** Build a position-level defensive rating model — how many points/rebounds/assists does this team allow to PG vs SG vs SF vs PF vs C? This dramatically improves prop accuracy.

4. **Bankroll Management Module:** Add Kelly Criterion or fractional Kelly stake sizing. Track historical pick performance, maintain a running P/L log, and auto-adjust confidence thresholds based on recent model accuracy.

5. **Multi-Game Slate Engine:** Extend the parlay builder to operate across an entire night's slate — automatically find the best 2-game or 3-game parlays mixing legs from different games for maximum diversification and reduced correlation.

---

## Bankroll / Risk Disclaimer

This tool provides **probabilistic analytics**, not investment advice. Key principles:

- **Never bet money you can't afford to lose.** Treat every dollar wagered as entertainment spending.
- **No model is perfect.** Even "High confidence" picks lose regularly. A 70% hit rate means 3 out of 10 bets lose.
- **Parlays are high-risk.** A 2-leg parlay at 60% per leg = ~36% combined. A 4-leg parlay = ~13%. The math works against you.
- **The house has an edge.** Sportsbooks set lines to profit. Finding true "edge" is rare and fleeting.
- **Track your results.** If you're not profitable after 100+ bets, reassess your approach.
- **$10/day max** is a reasonable bankroll discipline for experimentation.
