"""
Prediction interface.

Loads trained models and provides prediction functions for:
  - Player props (regression → projected stat + hit probability)
  - Game outcomes (classification → win probability)
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import date

from src.features.engineering import (
    build_player_feature_vector,
    build_game_feature_vector,
    percentile_estimate,
    hit_rate_at_line,
    _add_derived_stats,
)
from src.data.queries import player_recent, find_player_name
from src.utils.config import MODEL_DIR, resolve_team
from src.utils.logger import get_logger

log = get_logger(__name__)


def _load_model(name: str) -> dict | None:
    """Load a saved model bundle."""
    path = MODEL_DIR / f"{name}.joblib"
    if not path.exists():
        log.warning("Model not found: %s", path)
        return None
    return joblib.load(path)


def _features_to_df(feats: dict, feat_cols: list[str]) -> pd.DataFrame:
    """Convert feature dict to a single-row DataFrame with correct columns."""
    row = {c: feats.get(c, 0.0) for c in feat_cols}
    return pd.DataFrame([row])


# ---------------------------------------------------------------------------
# Confidence scoring
# ---------------------------------------------------------------------------

def confidence_tier(prob: float, sample_size: int, consistency: float) -> str:
    """
    Confidence scoring formula:
      score = probability * 0.5
            + min(sample_size / 20, 1.0) * 0.25     (data quality)
            + consistency * 0.25                      (player consistency)

    Tiers: High >= 0.65, Medium >= 0.45, Low < 0.45
    """
    data_q = min(sample_size / 20.0, 1.0)
    score = prob * 0.5 + data_q * 0.25 + consistency * 0.25
    if score >= 0.65:
        return "High"
    elif score >= 0.45:
        return "Medium"
    else:
        return "Low"


def confidence_score(prob: float, sample_size: int, consistency: float) -> float:
    """Numeric confidence score (0–1)."""
    data_q = min(sample_size / 20.0, 1.0)
    return prob * 0.5 + data_q * 0.25 + consistency * 0.25


# ---------------------------------------------------------------------------
# Player prop prediction
# ---------------------------------------------------------------------------

def predict_player_prop(player_name: str, stat_type: str, line: float | None,
                        opp_abbrev: str, game_date: date,
                        is_home: int = 1, over_under: str = "over") -> dict:
    """
    Predict a player prop.

    Returns: {
        projection, hit_probability, confidence_tier, confidence_score,
        reasons, risks, percentiles, hit_rates, data_quality
    }
    """
    # Resolve player name
    resolved = find_player_name(player_name)
    if not resolved:
        return {"error": f"Player not found: {player_name}"}
    player_name = resolved

    stat_col_map = {
        "points":   "points",
        "rebounds": "reboundsTotal",
        "assists":  "assists",
        "pra":      "pra",
        "pa":       "pa",
        "ra":       "ra",
        "threes":   "threePointersMade",
        "steals":   "steals",
        "blocks":   "blocks",
    }
    actual_col = stat_col_map.get(stat_type, stat_type)
    model_name_map = {
        "points":   "player_points",
        "rebounds": "player_rebounds",
        "assists":  "player_assists",
        "pra":      "player_pra",
        "pa":       "player_pa",
        "ra":       "player_ra",
        "threes":   "player_threes",
        "steals":   "player_steals",
        "blocks":   "player_blocks",
    }

    # Build features
    feats = build_player_feature_vector(
        player_name, opp_abbrev, game_date, is_home, stat_type, line
    )

    # Get recent data for analysis (add derived combo stats)
    recent = _add_derived_stats(player_recent(player_name, n=20))
    recent_vals = pd.to_numeric(recent[actual_col], errors="coerce").fillna(0).values if actual_col in recent.columns else np.array([])

    # Model prediction
    model_name = model_name_map.get(stat_type)
    projection = None
    model_bundle = _load_model(model_name) if model_name else None

    if model_bundle:
        feat_cols = model_bundle["meta"]["feature_cols"]
        X = _features_to_df(feats, feat_cols)
        projection = float(model_bundle["model"].predict(X)[0])
    else:
        # Fallback: use weighted recent average
        if len(recent_vals) > 0:
            weights = np.exp(-0.15 * np.arange(len(recent_vals)))
            projection = float(np.average(recent_vals, weights=weights))

    if projection is None:
        return {"error": "Not enough data for prediction"}

    # Hit probability estimation
    hit_prob = 0.5
    if line is not None and len(recent_vals) >= 5:
        # Combine model-based estimate with empirical hit rate.
        # Give more weight to empirical when sample is large (≥15 games).
        if over_under == "over":
            model_prob = _estimate_over_prob(projection, line, recent_vals)
            emp_rate = float((recent_vals > line).mean())
        else:
            model_prob = _estimate_under_prob(projection, line, recent_vals)
            emp_rate = float((recent_vals < line).mean())

        if len(recent_vals) >= 15:
            # Large sample: trust empirical more
            hit_prob = 0.4 * model_prob + 0.6 * emp_rate
        else:
            # Small sample: trust model more
            hit_prob = 0.6 * model_prob + 0.4 * emp_rate
    elif line is not None:
        if over_under == "over":
            hit_prob = _estimate_over_prob(projection, line, recent_vals)
        else:
            hit_prob = _estimate_under_prob(projection, line, recent_vals)

    # Projection std (used downstream for edge scoring)
    projection_std = float(recent_vals.std()) if len(recent_vals) >= 3 else 3.0

    # Consistency
    consistency = feats.get(f"{actual_col}_consistency",
                           feats.get("points_consistency", 0.5))

    sample = len(recent_vals)
    tier = confidence_tier(hit_prob, sample, consistency)
    score = confidence_score(hit_prob, sample, consistency)

    # Percentiles
    pctiles = percentile_estimate(recent, actual_col) if not recent.empty else {}

    # Reasons & risks
    reasons = _build_reasons(feats, projection, line, stat_type, over_under, recent_vals, opp_abbrev)
    risks = _build_risks(feats, projection, line, stat_type, recent_vals)

    result = {
        "player": player_name,
        "stat_type": stat_type,
        "line": line,
        "over_under": over_under,
        "projection": round(projection, 1),
        "projection_std": round(projection_std, 2),
        "hit_probability": round(hit_prob, 3),
        "confidence_tier": tier,
        "confidence_score": round(score, 3),
        "reasons": reasons,
        "risks": risks,
        "percentiles": {k: round(v, 1) for k, v in pctiles.items()},
        "recent_average": round(float(recent_vals[:10].mean()), 1) if len(recent_vals) >= 1 else None,
        "data_quality": _data_quality_notes(feats, sample),
        "suggested_usage": _suggested_usage(tier, hit_prob, consistency),
    }

    return result


def _estimate_over_prob(projection: float, line: float, recent_vals: np.ndarray) -> float:
    """Estimate probability of going OVER a line."""
    if len(recent_vals) < 3:
        return 0.55 if projection > line else 0.45
    std = max(recent_vals.std(), 1.0)
    # Use normal approximation
    z = (projection - line) / std
    from scipy.stats import norm
    return float(norm.cdf(z))


def _estimate_under_prob(projection: float, line: float, recent_vals: np.ndarray) -> float:
    return 1.0 - _estimate_over_prob(projection, line, recent_vals)


def _build_reasons(feats: dict, proj: float, line: float | None, stat_type: str,
                   over_under: str, recent_vals: np.ndarray, opp: str) -> list[str]:
    reasons = []
    if line is not None:
        diff = proj - line
        if over_under == "over" and diff > 0:
            reasons.append(f"Model projects {proj:.1f}, which is {diff:.1f} above the {line} line")
        elif over_under == "under" and diff < 0:
            reasons.append(f"Model projects {proj:.1f}, which is {abs(diff):.1f} below the {line} line")
        elif over_under == "over":
            reasons.append(f"Model projects {proj:.1f}, which is {abs(diff):.1f} BELOW the {line} line — risky over")
        else:
            reasons.append(f"Model projects {proj:.1f}, which is {diff:.1f} ABOVE the {line} line — risky under")

    if len(recent_vals) >= 5:
        avg5 = recent_vals[:5].mean()
        avg10 = recent_vals[:10].mean() if len(recent_vals) >= 10 else avg5
        if avg5 > avg10 * 1.05:
            reasons.append(f"Trending UP: last 5 avg {avg5:.1f} vs 10-game avg {avg10:.1f}")
        elif avg5 < avg10 * 0.95:
            reasons.append(f"Trending DOWN: last 5 avg {avg5:.1f} vs 10-game avg {avg10:.1f}")

    vs_sample = feats.get("vs_opp_sample", 0)
    if vs_sample >= 3:
        vs_avg = feats.get(f"vs_opp_{stat_type if stat_type != 'pra' else 'pra'}_avg", proj)
        reasons.append(f"Averages {vs_avg:.1f} in {vs_sample} games vs {opp}")

    if feats.get("is_b2b"):
        reasons.append("Playing on a back-to-back — could suppress output")

    if feats.get("likely_starter"):
        reasons.append("Projected starter (25+ min avg)")

    return reasons


def _build_risks(feats: dict, proj: float, line: float | None, stat_type: str,
                 recent_vals: np.ndarray) -> list[str]:
    risks = []
    if len(recent_vals) >= 5:
        std = recent_vals[:10].std()
        mean = recent_vals[:10].mean()
        if mean > 0 and std / mean > 0.35:
            risks.append(f"High variance: std/mean = {std/mean:.2f} — inconsistent performer for {stat_type}")

    if feats.get("minutes_std_10", 0) > 5:
        risks.append(f"Minutes volatility is high (std={feats.get('minutes_std_10', 0):.1f})")

    if feats.get("avg_recent_minutes", 30) < 22:
        risks.append("Low minutes recently — possible bench/role player risk")

    if feats.get("rest_days", 3) >= 7:
        risks.append("Extended rest — returning from absence, projection may be less reliable")

    if feats.get("games_played_30d", 10) < 5:
        risks.append("Limited recent games (possible injury/absence)—lower confidence")

    if feats.get("is_b2b"):
        risks.append("Back-to-back game — fatigue risk")

    return risks


def _data_quality_notes(feats: dict, sample: int) -> str:
    if sample >= 15:
        return "Good sample size (15+ recent games)"
    elif sample >= 8:
        return "Moderate sample (8–14 recent games)"
    elif sample >= 3:
        return "Small sample (3–7 games) — treat projection with caution"
    else:
        return "Very limited data — projection is unreliable"


def _suggested_usage(tier: str, prob: float, consistency: float) -> str:
    if tier == "High" and consistency > 0.6:
        return "parlay-safe"
    elif tier == "High":
        return "single or parlay"
    elif tier == "Medium":
        return "single only"
    else:
        return "aggressive only"


# ---------------------------------------------------------------------------
# Game outcome prediction
# ---------------------------------------------------------------------------

def predict_moneyline(home_abbrev: str, away_abbrev: str,
                      game_date: date) -> dict:
    """
    Predict moneyline outcome.

    Returns: {
        home_team, away_team, home_win_prob, away_win_prob,
        lean, confidence_tier, reasons, caution_flags
    }
    """
    feats = build_game_feature_vector(home_abbrev, away_abbrev, game_date)

    model_bundle = _load_model("moneyline")
    if model_bundle:
        feat_cols = model_bundle["meta"]["feature_cols"]
        X = _features_to_df(feats, feat_cols)
        home_prob = float(model_bundle["model"].predict_proba(X)[0][1])
    else:
        # Fallback: use simple feature-based heuristic
        home_score_avg = feats.get("home_team_teamScore_avg_10", 110)
        away_score_avg = feats.get("away_team_teamScore_avg_10", 110)
        margin = home_score_avg - away_score_avg + 3  # home court ~3 pts
        home_prob = min(max(0.5 + margin / 30, 0.25), 0.85)

    away_prob = 1.0 - home_prob

    if home_prob >= 0.55:
        lean = f"{home_abbrev} (Home)"
    elif away_prob >= 0.55:
        lean = f"{away_abbrev} (Away)"
    else:
        lean = "Toss-up"

    h2h = feats.get("h2h_sample", 0)
    tier = "High" if abs(home_prob - 0.5) > 0.15 and h2h >= 3 else \
           "Medium" if abs(home_prob - 0.5) > 0.08 else "Low"

    reasons = []
    for prefix, team in [("home", home_abbrev), ("away", away_abbrev)]:
        wr = feats.get(f"{prefix}_team_win_rate_10", 0.5)
        reasons.append(f"{team} last-10 win rate: {wr:.0%}")
        score = feats.get(f"{prefix}_team_teamScore_avg_10", 0)
        opp_score = feats.get(f"{prefix}_team_opponentScore_avg_10", 0)
        if score and opp_score:
            reasons.append(f"{team} avg margin (10g): {score - opp_score:+.1f}")

    caution = []
    if abs(home_prob - 0.5) < 0.08:
        caution.append("Very close matchup — high variance expected")

    home_rest = feats.get("home_team_rest_days", 2)
    away_rest = feats.get("away_team_rest_days", 2)
    if home_rest <= 1:
        caution.append(f"{home_abbrev} on back-to-back")
    if away_rest <= 1:
        caution.append(f"{away_abbrev} on back-to-back")

    # Blowout risk
    margin_diff = abs(feats.get("diff_teamScore_10", 0))
    if margin_diff > 10:
        caution.append("Significant talent gap — blowout risk (props may be fragile)")

    return {
        "home_team": home_abbrev,
        "away_team": away_abbrev,
        "home_win_probability": round(home_prob, 3),
        "away_win_probability": round(away_prob, 3),
        "lean": lean,
        "projected_margin": round(home_prob * 15 - 7.5, 1),  # rough spread estimate
        "confidence_tier": tier,
        "reasons": reasons,
        "caution_flags": caution,
    }


def suggest_player_props(player_name: str, opp_abbrev: str,
                         game_date: date, is_home: int = 1,
                         risk_mode: str = "balanced") -> list[dict]:
    """
    Suggest best prop lines for a player.

    Returns a list of prop suggestions sorted by confidence.
    """
    resolved = find_player_name(player_name)
    if not resolved:
        return [{"error": f"Player not found: {player_name}"}]
    player_name = resolved

    stat_types = ["points", "rebounds", "assists", "pra", "pa", "ra",
                  "threes", "steals", "blocks"]
    suggestions = []

    for st in stat_types:
        # First predict without a line to get projection
        result = predict_player_prop(
            player_name, st, line=None, opp_abbrev=opp_abbrev,
            game_date=game_date, is_home=is_home
        )
        if "error" in result:
            continue

        proj = result["projection"]
        if proj is None or proj <= 0:
            continue

        # Suggest lines snapped to realistic 0.5 sportsbook increments
        if risk_mode == "safe":
            # Conservative: set line ~10% below projection for cushion
            raw = proj * 0.88
        elif risk_mode == "aggressive":
            # Aggressive: line slightly above projection for upside
            raw = proj * 1.05
        else:
            # Balanced: line just below projection
            raw = proj * 0.94
        # Snap to nearest 0.5 increment
        line = round(raw * 2) / 2
        direction = "over"

        # Re-predict with the suggested line
        with_line = predict_player_prop(
            player_name, st, line=line, opp_abbrev=opp_abbrev,
            game_date=game_date, is_home=is_home, over_under=direction
        )
        if "error" not in with_line:
            with_line["suggested_line"] = line
            with_line["direction"] = direction
            suggestions.append(with_line)

    # Sort by confidence score
    suggestions.sort(key=lambda x: x.get("confidence_score", 0), reverse=True)
    return suggestions
