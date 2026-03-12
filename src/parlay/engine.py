"""
Parlay Engine.

Assembles, scores, filters, and ranks parlay combinations.

Key concepts:
  - Each "leg" is a single bet (player prop or moneyline)
  - Parlays combine legs with combined hit probability
  - Correlation penalty reduces estimated combined probability when legs
    depend on each other (e.g., same game, same team)
  - Risk modes: safe, balanced, aggressive
"""

import itertools
from dataclasses import dataclass, field
from datetime import date

from src.models.predict import (
    predict_player_prop, predict_moneyline, suggest_player_props,
    confidence_score,
)
from src.utils.config import resolve_team
from src.utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Leg:
    """A single bet leg."""
    leg_type: str               # "player_prop" or "moneyline"
    description: str            # e.g. "LeBron James Over 27.5 Points"
    player: str | None = None
    stat_type: str | None = None
    line: float | None = None
    over_under: str | None = None
    team: str | None = None
    opponent: str | None = None
    game_date: date | None = None
    hit_probability: float = 0.5
    confidence_tier: str = "Medium"
    confidence_score: float = 0.5
    reasons: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    projection: float | None = None
    suggested_usage: str = "single only"
    game_key: str = ""          # "HOME_AWAY" for correlation detection


@dataclass
class Parlay:
    """A multi-leg parlay."""
    name: str
    mode: str                   # safe / balanced / aggressive
    legs: list[Leg]
    combined_probability: float
    correlation_penalty: float
    adjusted_probability: float
    reasoning: str
    warnings: list[str] = field(default_factory=list)
    alternatives: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Correlation penalty
# ---------------------------------------------------------------------------

def correlation_penalty(legs: list[Leg], allow_correlation: bool = False) -> float:
    """
    Compute a correlation penalty for a set of legs.

    Formula:
      base_penalty = 0.0
      For each pair of legs in the same game:
        base_penalty += 0.04
      For each pair from the same team (in the same game):
        base_penalty += 0.06
      For each pair involving the same player:
        base_penalty += 0.10  (shouldn't happen, but guard)

    If allow_correlation is True, penalty is halved.

    Returns a multiplier (0–1) to apply to combined probability.
    """
    penalty = 0.0
    for i, j in itertools.combinations(range(len(legs)), 2):
        a, b = legs[i], legs[j]
        # Same game
        if a.game_key and b.game_key and a.game_key == b.game_key:
            penalty += 0.04
            # Same team within game
            if a.team and b.team and a.team == b.team:
                penalty += 0.06
        # Same player (guard)
        if a.player and b.player and a.player == b.player:
            penalty += 0.10

    if allow_correlation:
        penalty *= 0.5

    return min(penalty, 0.40)  # Cap at 40% penalty


def combined_hit_probability(legs: list[Leg], allow_corr: bool = False) -> tuple[float, float, float]:
    """
    Calculate combined parlay probability.

    Returns: (raw_combined, penalty, adjusted_combined)
    """
    if not legs:
        return 0.0, 0.0, 0.0

    raw = 1.0
    for leg in legs:
        raw *= leg.hit_probability

    pen = correlation_penalty(legs, allow_corr)
    adjusted = raw * (1.0 - pen)

    return raw, pen, adjusted


# ---------------------------------------------------------------------------
# Leg filtering
# ---------------------------------------------------------------------------

def filter_legs(legs: list[Leg], constraints: dict | None = None) -> list[Leg]:
    """
    Filter legs based on constraints.

    Constraints:
      min_confidence: float (0–1), default 0.45
      only_player_props: bool
      only_one_per_player: bool
      avoid_bench: bool (exclude players with < 25 avg minutes)
      avoid_low_minutes: bool
      stat_types: list[str] or None
      max_legs_per_game: int
    """
    if constraints is None:
        constraints = {}

    min_conf = constraints.get("min_confidence", 0.45)
    only_props = constraints.get("only_player_props", False)
    one_per_player = constraints.get("only_one_per_player", True)
    avoid_bench = constraints.get("avoid_bench", True)
    stat_types = constraints.get("stat_types", None)

    filtered = []
    seen_players = set()

    for leg in legs:
        # Confidence threshold
        if leg.confidence_score < min_conf:
            continue

        # Only player props
        if only_props and leg.leg_type != "player_prop":
            continue

        # One per player
        if one_per_player and leg.player:
            if leg.player in seen_players:
                continue
            seen_players.add(leg.player)

        # Avoid bench
        if avoid_bench and leg.suggested_usage == "aggressive only":
            continue

        # Stat type filter
        if stat_types and leg.stat_type and leg.stat_type not in stat_types:
            continue

        filtered.append(leg)

    return filtered


# ---------------------------------------------------------------------------
# Parlay builder
# ---------------------------------------------------------------------------

RISK_CONFIGS = {
    "safe": {
        "min_confidence": 0.55,
        "avoid_bench": True,
        "only_one_per_player": True,
        "prefer_high_consistency": True,
        "max_legs": 2,
    },
    "balanced": {
        "min_confidence": 0.45,
        "avoid_bench": True,
        "only_one_per_player": True,
        "prefer_high_consistency": False,
        "max_legs": 3,
    },
    "aggressive": {
        "min_confidence": 0.35,
        "avoid_bench": False,
        "only_one_per_player": False,
        "prefer_high_consistency": False,
        "max_legs": 6,
    },
}


def build_legs_for_game(home_abbrev: str, away_abbrev: str,
                        game_date: date,
                        players: list[str] | None = None,
                        risk_mode: str = "balanced") -> list[Leg]:
    """
    Generate candidate legs for a single game.

    If players is None, we can't auto-discover players from the schedule,
    so we return the moneyline leg only. Provide player names for prop legs.
    """
    legs: list[Leg] = []
    game_key = f"{home_abbrev}_{away_abbrev}"

    # Moneyline leg
    ml = predict_moneyline(home_abbrev, away_abbrev, game_date)
    if "error" not in ml:
        fav = home_abbrev if ml["home_win_probability"] > 0.5 else away_abbrev
        prob = max(ml["home_win_probability"], ml["away_win_probability"])
        legs.append(Leg(
            leg_type="moneyline",
            description=f"{fav} ML",
            team=fav,
            opponent=away_abbrev if fav == home_abbrev else home_abbrev,
            game_date=game_date,
            hit_probability=prob,
            confidence_tier=ml["confidence_tier"],
            confidence_score=confidence_score(prob, 20, 0.6),
            reasons=ml["reasons"],
            risks=ml.get("caution_flags", []),
            projection=ml.get("projected_margin"),
            suggested_usage="parlay-safe" if prob > 0.6 else "single only",
            game_key=game_key,
        ))

    # Player prop legs
    if players:
        for pname in players:
            team_abbrev = None
            # Try to figure out which team this player is on
            from src.data.queries import player_recent
            rec = player_recent(pname, n=1)
            if not rec.empty:
                team_abbrev = rec.iloc[0].get("team_abbrev", "")

            is_home = 1 if team_abbrev == home_abbrev else 0
            opp = away_abbrev if is_home else home_abbrev

            props = suggest_player_props(pname, opp, game_date, is_home, risk_mode)
            for prop in props:
                if "error" in prop:
                    continue
                legs.append(Leg(
                    leg_type="player_prop",
                    description=f"{prop['player']} {prop.get('direction', 'Over')} {prop.get('suggested_line', prop.get('line', '?'))} {prop['stat_type']}",
                    player=prop["player"],
                    stat_type=prop["stat_type"],
                    line=prop.get("suggested_line", prop.get("line")),
                    over_under=prop.get("direction", "over"),
                    team=team_abbrev,
                    opponent=opp,
                    game_date=game_date,
                    hit_probability=prop["hit_probability"],
                    confidence_tier=prop["confidence_tier"],
                    confidence_score=prop["confidence_score"],
                    reasons=prop.get("reasons", []),
                    risks=prop.get("risks", []),
                    projection=prop.get("projection"),
                    suggested_usage=prop.get("suggested_usage", "single only"),
                    game_key=game_key,
                ))

    return legs


def make_parlay(legs: list[Leg], num_legs: int = 2,
                risk_mode: str = "balanced",
                allow_correlation: bool = False,
                constraints: dict | None = None) -> list[Parlay]:
    """
    Build the best parlays from candidate legs.

    Returns ranked list of parlays.
    """
    config = RISK_CONFIGS.get(risk_mode, RISK_CONFIGS["balanced"])
    if constraints is None:
        constraints = {}

    # Merge config constraints with user constraints
    merged = {**config, **constraints}
    max_legs = merged.get("max_legs", num_legs)
    actual_legs = min(num_legs, max_legs)

    # Filter legs
    filtered = filter_legs(legs, merged)
    if len(filtered) < actual_legs:
        # Relax constraints
        relaxed = {**merged, "min_confidence": max(merged.get("min_confidence", 0.45) - 0.10, 0.30)}
        filtered = filter_legs(legs, relaxed)

    if len(filtered) < actual_legs:
        return []

    # Sort by confidence
    filtered.sort(key=lambda x: x.confidence_score, reverse=True)

    # Generate combinations
    parlays: list[Parlay] = []
    seen = set()

    for combo in itertools.combinations(filtered, actual_legs):
        combo_list = list(combo)

        # Deduplicate by description set
        desc_key = frozenset(l.description for l in combo_list)
        if desc_key in seen:
            continue
        seen.add(desc_key)

        raw, pen, adj = combined_hit_probability(combo_list, allow_correlation)

        # Skip very low probability parlays
        if adj < 0.05:
            continue

        warnings = []
        if pen > 0.10:
            warnings.append(f"Correlation penalty of {pen:.0%} applied — legs may be dependent")

        # Check for blowout risk
        for l in combo_list:
            if any("blowout" in r.lower() for r in l.risks):
                warnings.append("Blowout risk in at least one game — props may bust")
                break

        reasoning = _parlay_reasoning(combo_list, risk_mode)

        parlays.append(Parlay(
            name=f"{risk_mode.title()} {actual_legs}-Leg Parlay",
            mode=risk_mode,
            legs=combo_list,
            combined_probability=round(raw, 4),
            correlation_penalty=round(pen, 4),
            adjusted_probability=round(adj, 4),
            reasoning=reasoning,
            warnings=warnings,
        ))

        if len(parlays) >= 5:  # Return top 5
            break

    # Sort by adjusted probability
    parlays.sort(key=lambda p: p.adjusted_probability, reverse=True)
    return parlays


def _parlay_reasoning(legs: list[Leg], mode: str) -> str:
    parts = []
    if mode == "safe":
        parts.append("Selected high-confidence, high-consistency legs.")
    elif mode == "aggressive":
        parts.append("Selected higher-upside legs with more variance.")
    else:
        parts.append("Balanced selection mixing confidence with value.")

    teams = {l.game_key for l in legs if l.game_key}
    if len(teams) > 1:
        parts.append("Spread across multiple games for diversification.")
    elif len(teams) == 1:
        parts.append("Same-game parlay — correlated outcomes possible.")

    return " ".join(parts)


# ---------------------------------------------------------------------------
# High-level parlay helpers
# ---------------------------------------------------------------------------

def find_best_legs(all_legs: list[Leg], risk_mode: str = "balanced",
                   stat_types: list[str] | None = None,
                   min_confidence: float = 0.45,
                   top_n: int = 10) -> list[Leg]:
    """Find the top candidate legs ranked by confidence."""
    constraints = {
        "min_confidence": min_confidence,
        "stat_types": stat_types,
        "only_one_per_player": True,
        "avoid_bench": risk_mode != "aggressive",
    }
    filtered = filter_legs(all_legs, constraints)
    filtered.sort(key=lambda x: x.confidence_score, reverse=True)
    return filtered[:top_n]


def fade_risky_legs(all_legs: list[Leg], top_n: int = 5) -> list[dict]:
    """Find legs to AVOID and explain why."""
    risky = []
    for leg in all_legs:
        risk_score = 0
        if leg.confidence_score < 0.40:
            risk_score += 2
        if "inconsistent" in " ".join(leg.risks).lower():
            risk_score += 1
        if "bench" in " ".join(leg.risks).lower() or "low minutes" in " ".join(leg.risks).lower():
            risk_score += 2
        if "back-to-back" in " ".join(leg.risks).lower():
            risk_score += 1
        if "absence" in " ".join(leg.risks).lower():
            risk_score += 2
        if "blowout" in " ".join(leg.risks).lower():
            risk_score += 1

        if risk_score >= 2:
            risky.append({
                "leg": leg.description,
                "risk_score": risk_score,
                "reasons_to_avoid": leg.risks,
                "hit_probability": leg.hit_probability,
                "confidence_tier": leg.confidence_tier,
            })

    risky.sort(key=lambda x: x["risk_score"], reverse=True)
    return risky[:top_n]


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def leg_to_dict(leg: Leg) -> dict:
    return {
        "type": leg.leg_type,
        "description": leg.description,
        "player": leg.player,
        "stat_type": leg.stat_type,
        "line": leg.line,
        "over_under": leg.over_under,
        "team": leg.team,
        "opponent": leg.opponent,
        "projection": leg.projection,
        "hit_probability": leg.hit_probability,
        "confidence_tier": leg.confidence_tier,
        "confidence_score": leg.confidence_score,
        "reasons": leg.reasons,
        "risks": leg.risks,
        "suggested_usage": leg.suggested_usage,
    }


def parlay_to_dict(parlay: Parlay) -> dict:
    return {
        "name": parlay.name,
        "mode": parlay.mode,
        "legs": [leg_to_dict(l) for l in parlay.legs],
        "combined_probability": parlay.combined_probability,
        "correlation_penalty": parlay.correlation_penalty,
        "adjusted_probability": parlay.adjusted_probability,
        "reasoning": parlay.reasoning,
        "warnings": parlay.warnings,
        "disclaimer": (
            "DISCLAIMER: These are probabilistic analytics, NOT guarantees. "
            "Sports betting carries inherent risk. Never bet more than you can "
            "afford to lose. Past performance does not predict future results. "
            "This tool provides data-driven analysis, not financial advice."
        ),
    }
