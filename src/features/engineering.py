"""
Feature engineering for player props and team/game predictions.

Builds rolling-window, split, and matchup features from the DuckDB tables.
All features are built from data strictly BEFORE the target game date to
prevent leakage.
"""

import numpy as np
import pandas as pd
from datetime import date, timedelta

from src.data.ingest import get_connection
from src.utils.logger import get_logger

log = get_logger(__name__)

STAT_COLS = ["points", "reboundsTotal", "assists", "pra",
             "threePointersMade", "steals", "blocks", "numMinutes",
             "fieldGoalsMade", "fieldGoalsAttempted",
             "freeThrowsMade", "freeThrowsAttempted"]

TEAM_STAT_COLS = ["teamScore", "opponentScore", "assists", "reboundsTotal",
                  "turnovers", "fieldGoalsMade", "fieldGoalsAttempted",
                  "threePointersMade", "threePointersAttempted",
                  "freeThrowsMade", "freeThrowsAttempted",
                  "reboundsOffensive", "reboundsDefensive"]


# ---------------------------------------------------------------------------
# Player features
# ---------------------------------------------------------------------------

def player_rolling_features(game_log: pd.DataFrame,
                            windows: list[int] = [3, 5, 10]) -> dict:
    """
    Compute rolling averages, medians, std devs, and hit-rate style features
    from a player game log sorted by date descending.

    Returns a flat dict of feature_name -> value.
    """
    if game_log.empty:
        return {}

    feats: dict = {}
    gl = game_log.sort_values("game_date", ascending=False).copy()

    for col in STAT_COLS:
        if col not in gl.columns:
            continue
        vals = pd.to_numeric(gl[col], errors="coerce").fillna(0)
        for w in windows:
            subset = vals.head(w)
            if len(subset) == 0:
                continue
            feats[f"{col}_avg_{w}"] = float(subset.mean())
            feats[f"{col}_med_{w}"] = float(subset.median())
            feats[f"{col}_std_{w}"] = float(subset.std()) if len(subset) > 1 else 0.0
            feats[f"{col}_min_{w}"] = float(subset.min())
            feats[f"{col}_max_{w}"] = float(subset.max())

    # Weighted recent form: exponential decay weights over last 10
    for col in ["points", "reboundsTotal", "assists", "pra"]:
        if col not in gl.columns:
            continue
        vals = pd.to_numeric(gl[col].head(10), errors="coerce").fillna(0).values
        if len(vals) == 0:
            continue
        weights = np.exp(-0.15 * np.arange(len(vals)))
        feats[f"{col}_ewm"] = float(np.average(vals, weights=weights))

    # Minutes trend
    if "numMinutes" in gl.columns:
        mins = pd.to_numeric(gl["numMinutes"].head(10), errors="coerce").fillna(0).values
        if len(mins) >= 3:
            feats["minutes_trend_3"] = float(mins[:3].mean() - mins.mean())
            feats["minutes_avg_10"] = float(mins.mean())
            feats["minutes_std_10"] = float(mins.std())

    # Consistency score = 1 - (CV)  (lower variance vs mean = more consistent)
    for col in ["points", "pra"]:
        if col not in gl.columns:
            continue
        vals = pd.to_numeric(gl[col].head(10), errors="coerce").fillna(0)
        mn = vals.mean()
        if mn > 0:
            feats[f"{col}_consistency"] = float(1.0 - min(vals.std() / mn, 1.0))
        else:
            feats[f"{col}_consistency"] = 0.0

    # Games played (to detect absences)
    cutoff_30d = pd.Timestamp.now() - timedelta(days=30)
    gd_series = pd.to_datetime(gl["game_date"], errors="coerce")
    feats["games_played_30d"] = int((gd_series >= cutoff_30d).sum())

    # Home/away splits from last 20
    if "home" in gl.columns:
        for ha, label in [(1, "home"), (0, "away")]:
            sub = gl[gl["home"] == ha].head(10)
            for col in ["points", "reboundsTotal", "assists", "pra"]:
                if col in sub.columns:
                    vals = pd.to_numeric(sub[col], errors="coerce").fillna(0)
                    feats[f"{col}_{label}_avg"] = float(vals.mean()) if len(vals) > 0 else 0.0

    return feats


def player_opponent_features(vs_log: pd.DataFrame) -> dict:
    """Features from a player's history against a specific opponent."""
    if vs_log.empty or len(vs_log) < 2:
        return {"vs_opp_sample": 0}

    feats: dict = {"vs_opp_sample": len(vs_log)}
    for col in ["points", "reboundsTotal", "assists", "pra"]:
        if col not in vs_log.columns:
            continue
        vals = pd.to_numeric(vs_log[col], errors="coerce").fillna(0)
        feats[f"vs_opp_{col}_avg"] = float(vals.mean())
        feats[f"vs_opp_{col}_med"] = float(vals.median())
    return feats


def player_rest_features(game_log: pd.DataFrame, target_date: date) -> dict:
    """Rest days and back-to-back flag."""
    if game_log.empty:
        return {"rest_days": 3, "is_b2b": 0}

    gl = game_log.sort_values("game_date", ascending=False)
    last = pd.Timestamp(gl.iloc[0]["game_date"]).date()
    rest = (target_date - last).days
    is_b2b = 1 if rest <= 1 else 0
    return {"rest_days": rest, "is_b2b": is_b2b}


def player_starter_signal(game_log: pd.DataFrame) -> dict:
    """Infer starter vs bench from recent minutes."""
    if game_log.empty:
        return {"likely_starter": 0, "avg_recent_minutes": 0.0}
    mins = pd.to_numeric(game_log["numMinutes"].head(5), errors="coerce").fillna(0)
    avg_min = float(mins.mean())
    return {
        "likely_starter": 1 if avg_min >= 25 else 0,
        "avg_recent_minutes": avg_min,
    }


def hit_rate_at_line(game_log: pd.DataFrame, stat_col: str, line: float,
                     windows: list[int] = [5, 10, 20]) -> dict:
    """Fraction of recent games where player exceeded a line."""
    feats = {}
    if game_log.empty or stat_col not in game_log.columns:
        return feats
    vals = pd.to_numeric(game_log[stat_col], errors="coerce").fillna(0)
    for w in windows:
        sub = vals.head(w)
        if len(sub) == 0:
            continue
        feats[f"hit_rate_{stat_col}_{line}_{w}"] = float((sub > line).mean())
    return feats


def percentile_estimate(game_log: pd.DataFrame, stat_col: str) -> dict:
    """Return percentile markers for a stat from recent games."""
    if game_log.empty or stat_col not in game_log.columns:
        return {}
    vals = pd.to_numeric(game_log[stat_col].head(20), errors="coerce").fillna(0).values
    if len(vals) < 3:
        return {}
    return {
        f"{stat_col}_p10": float(np.percentile(vals, 10)),
        f"{stat_col}_p25": float(np.percentile(vals, 25)),
        f"{stat_col}_p50": float(np.percentile(vals, 50)),
        f"{stat_col}_p75": float(np.percentile(vals, 75)),
        f"{stat_col}_p90": float(np.percentile(vals, 90)),
    }


# ---------------------------------------------------------------------------
# Team features
# ---------------------------------------------------------------------------

def team_rolling_features(team_log: pd.DataFrame,
                          windows: list[int] = [5, 10]) -> dict:
    """Rolling team performance features."""
    if team_log.empty:
        return {}

    feats: dict = {}
    tl = team_log.sort_values("game_date", ascending=False).copy()

    for col in TEAM_STAT_COLS:
        if col not in tl.columns:
            continue
        vals = pd.to_numeric(tl[col], errors="coerce").fillna(0)
        for w in windows:
            sub = vals.head(w)
            if len(sub) == 0:
                continue
            feats[f"team_{col}_avg_{w}"] = float(sub.mean())

    # Win rate
    if "win" in tl.columns:
        for w in windows:
            sub = tl["win"].head(w)
            feats[f"team_win_rate_{w}"] = float(sub.mean())

    # Home/Away
    if "home" in tl.columns:
        home_g = tl[tl["home"] == 1].head(10)
        away_g = tl[tl["home"] == 0].head(10)
        if not home_g.empty:
            feats["team_home_score_avg"] = float(pd.to_numeric(home_g["teamScore"], errors="coerce").mean())
        if not away_g.empty:
            feats["team_away_score_avg"] = float(pd.to_numeric(away_g["teamScore"], errors="coerce").mean())

    # Pace proxy = team FGA + 0.44*FTA - OREB + TOV  (per-game approx)
    for w in [5, 10]:
        sub = tl.head(w)
        try:
            fga = pd.to_numeric(sub["fieldGoalsAttempted"], errors="coerce").fillna(0)
            fta = pd.to_numeric(sub["freeThrowsAttempted"], errors="coerce").fillna(0)
            oreb = pd.to_numeric(sub.get("reboundsOffensive", 0), errors="coerce").fillna(0)
            tov = pd.to_numeric(sub["turnovers"], errors="coerce").fillna(0)
            pace = fga + 0.44 * fta - oreb + tov
            feats[f"team_pace_proxy_{w}"] = float(pace.mean())
        except Exception:
            pass

    return feats


def team_defensive_features(opp_team_log: pd.DataFrame,
                            windows: list[int] = [5, 10]) -> dict:
    """How much do opponents score against this team?"""
    if opp_team_log.empty:
        return {}

    feats: dict = {}
    tl = opp_team_log.sort_values("game_date", ascending=False)

    for w in windows:
        sub = tl.head(w)
        opp_pts = pd.to_numeric(sub.get("opponentScore", pd.Series()), errors="coerce").fillna(0)
        if len(opp_pts) > 0:
            feats[f"team_def_pts_allowed_{w}"] = float(opp_pts.mean())
        team_pts = pd.to_numeric(sub.get("teamScore", pd.Series()), errors="coerce").fillna(0)
        if len(team_pts) > 0 and len(opp_pts) > 0:
            feats[f"team_def_margin_{w}"] = float(team_pts.mean() - opp_pts.mean())

    return feats


def team_rest_features(team_log: pd.DataFrame, target_date: date) -> dict:
    """Team rest days and back-to-back."""
    if team_log.empty:
        return {"team_rest_days": 3, "team_is_b2b": 0}
    tl = team_log.sort_values("game_date", ascending=False)
    last = pd.Timestamp(tl.iloc[0]["game_date"]).date()
    rest = (target_date - last).days
    return {"team_rest_days": rest, "team_is_b2b": 1 if rest <= 1 else 0}


# ---------------------------------------------------------------------------
# Full feature vector builders
# ---------------------------------------------------------------------------

def build_player_feature_vector(player_name: str, opp_abbrev: str,
                                 target_date: date, is_home: int,
                                 stat_type: str = "points",
                                 line: float | None = None) -> dict:
    """Build the complete feature dict for a player prop prediction."""
    from src.data.queries import player_recent, player_vs_opponent, player_game_log

    # Get game logs
    recent = player_recent(player_name, n=30)
    vs_opp = player_vs_opponent(player_name, opp_abbrev, n=10)

    feats: dict = {"is_home": is_home}

    # Rolling features
    feats.update(player_rolling_features(recent))

    # Opponent features
    feats.update(player_opponent_features(vs_opp))

    # Rest
    feats.update(player_rest_features(recent, target_date))

    # Starter signal
    feats.update(player_starter_signal(recent))

    # Percentiles for target stat
    stat_col_map = {
        "points": "points", "rebounds": "reboundsTotal",
        "assists": "assists", "pra": "pra",
        "threes": "threePointersMade", "steals": "steals", "blocks": "blocks",
    }
    actual_col = stat_col_map.get(stat_type, stat_type)
    feats.update(percentile_estimate(recent, actual_col))

    # Hit rate if line provided
    if line is not None:
        feats.update(hit_rate_at_line(recent, actual_col, line))

    return feats


def build_game_feature_vector(home_abbrev: str, away_abbrev: str,
                               target_date: date) -> dict:
    """Build feature dict for a game-level (moneyline/spread) prediction."""
    from src.data.queries import team_recent, team_vs_team

    home_log = team_recent(home_abbrev, n=20)
    away_log = team_recent(away_abbrev, n=20)
    h2h_home = team_vs_team(home_abbrev, away_abbrev, n=10)
    h2h_away = team_vs_team(away_abbrev, home_abbrev, n=10)

    feats: dict = {}

    # Home team features
    home_feats = team_rolling_features(home_log)
    feats.update({f"home_{k}": v for k, v in home_feats.items()})
    feats.update({f"home_{k}": v for k, v in team_defensive_features(home_log).items()})
    feats.update({f"home_{k}": v for k, v in team_rest_features(home_log, target_date).items()})

    # Away team features
    away_feats = team_rolling_features(away_log)
    feats.update({f"away_{k}": v for k, v in away_feats.items()})
    feats.update({f"away_{k}": v for k, v in team_defensive_features(away_log).items()})
    feats.update({f"away_{k}": v for k, v in team_rest_features(away_log, target_date).items()})

    # H2H
    if not h2h_home.empty:
        home_wins_h2h = h2h_home["win"].sum() if "win" in h2h_home.columns else 0
        feats["h2h_home_win_rate"] = float(home_wins_h2h / len(h2h_home))
        feats["h2h_sample"] = len(h2h_home)
    else:
        feats["h2h_home_win_rate"] = 0.5
        feats["h2h_sample"] = 0

    # Differentials
    for col in ["teamScore", "opponentScore", "assists", "reboundsTotal"]:
        hk = f"home_team_{col}_avg_10"
        ak = f"away_team_{col}_avg_10"
        if hk in feats and ak in feats:
            feats[f"diff_{col}_10"] = feats[hk] - feats[ak]

    return feats
