"""
MCP Tool definitions for NBA Parlay Maker.

Each tool is a function that takes structured input and returns
structured output suitable for Claude / MCP client consumption.
"""

import json
from datetime import date, datetime

from src.models.predict import (
    predict_player_prop, predict_moneyline,
    suggest_player_props, confidence_score,
)
from src.parlay.engine import (
    build_legs_for_game, make_parlay, find_best_legs,
    fade_risky_legs, leg_to_dict, parlay_to_dict, Leg,
)
from src.data.queries import (
    player_recent, team_recent, find_player_name,
    games_on_date, schedule_on_date,
)
from src.utils.config import resolve_team
from src.utils.logger import get_logger

log = get_logger(__name__)

DISCLAIMER = (
    "DISCLAIMER: These are probabilistic analytics based on historical data, "
    "NOT guarantees. Sports betting carries inherent risk. Never bet more than "
    "you can afford to lose. Past performance does not predict future results."
)


def _parse_date(d: str | None) -> date:
    """Parse a date string, default to today."""
    if not d:
        return date.today()
    try:
        return datetime.strptime(d, "%Y-%m-%d").date()
    except Exception:
        return date.today()


# ---------------------------------------------------------------------------
# Tool: analyze_game
# ---------------------------------------------------------------------------

def analyze_game(home_team: str, away_team: str, game_date: str | None = None) -> dict:
    """
    Analyze a game matchup: projected winner, confidence, key reasons, caution flags.

    Args:
        home_team: Home team name (e.g. "Lakers", "LAL", "Los Angeles Lakers")
        away_team: Away team name
        game_date: Date in YYYY-MM-DD format (defaults to today)
    """
    home = resolve_team(home_team)
    away = resolve_team(away_team)
    if not home:
        return {"error": f"Could not resolve home team: {home_team}"}
    if not away:
        return {"error": f"Could not resolve away team: {away_team}"}

    gd = _parse_date(game_date)
    ml = predict_moneyline(home, away, gd)

    # Enrich with recent form
    home_recent = team_recent(home, 10)
    away_recent = team_recent(away, 10)

    summary = {
        "game": f"{home} vs {away}",
        "date": str(gd),
        **ml,
        "home_recent_record": None,
        "away_recent_record": None,
        "disclaimer": DISCLAIMER,
    }

    if not home_recent.empty and "win" in home_recent.columns:
        hw = int(home_recent["win"].sum())
        summary["home_recent_record"] = f"{hw}-{len(home_recent) - hw} (last 10)"
    if not away_recent.empty and "win" in away_recent.columns:
        aw = int(away_recent["win"].sum())
        summary["away_recent_record"] = f"{aw}-{len(away_recent) - aw} (last 10)"

    return summary


# ---------------------------------------------------------------------------
# Tool: evaluate_moneyline
# ---------------------------------------------------------------------------

def evaluate_moneyline(home_team: str, away_team: str,
                       game_date: str | None = None) -> dict:
    """
    Evaluate moneyline bet: win probabilities, lean, explanation.

    Args:
        home_team: Home team name
        away_team: Away team name
        game_date: YYYY-MM-DD (defaults to today)
    """
    home = resolve_team(home_team)
    away = resolve_team(away_team)
    if not home or not away:
        return {"error": f"Could not resolve teams: {home_team} / {away_team}"}

    gd = _parse_date(game_date)
    result = predict_moneyline(home, away, gd)
    result["disclaimer"] = DISCLAIMER
    return result


# ---------------------------------------------------------------------------
# Tool: evaluate_player_prop
# ---------------------------------------------------------------------------

def evaluate_player_prop(player_name: str, stat_type: str,
                         line: float | None = None,
                         opponent: str | None = None,
                         game_date: str | None = None,
                         over_under: str = "over") -> dict:
    """
    Evaluate a player prop bet.

    Args:
        player_name: Full or partial player name (e.g. "LeBron", "Jokic")
        stat_type: One of: points, rebounds, assists, pra, threes, steals, blocks
        line: The betting line (e.g. 27.5). If None, returns projection and suggested lines.
        opponent: Opponent team name (e.g. "Celtics")
        game_date: YYYY-MM-DD (defaults to today)
        over_under: "over" or "under" (defaults to "over")
    """
    opp_abbrev = resolve_team(opponent) if opponent else ""
    if opponent and not opp_abbrev:
        return {"error": f"Could not resolve opponent: {opponent}"}

    gd = _parse_date(game_date)

    # Try to determine if home
    is_home = 1  # default
    if opp_abbrev:
        # Check schedule
        sched = schedule_on_date(str(gd))
        if not sched.empty:
            resolved_name = find_player_name(player_name)
            if resolved_name:
                rec = player_recent(resolved_name, 1)
                if not rec.empty:
                    player_team = rec.iloc[0].get("team_abbrev", "")
                    # Check if home or away
                    for _, row in sched.iterrows():
                        h = str(row.get("homeTeamName", row.get("hometeamName", ""))).strip()
                        if player_team and player_team in str(row.get("homeTeamId", "")):
                            is_home = 1
                            break
                        elif player_team and player_team in str(row.get("awayTeamId", "")):
                            is_home = 0
                            break

    result = predict_player_prop(
        player_name, stat_type, line, opp_abbrev or "", gd, is_home, over_under
    )
    result["disclaimer"] = DISCLAIMER
    return result


# ---------------------------------------------------------------------------
# Tool: suggest_player_props
# ---------------------------------------------------------------------------

def suggest_player_props_tool(player_name: str, opponent: str | None = None,
                              game_date: str | None = None,
                              risk_mode: str = "balanced") -> dict:
    """
    Suggest the best prop bets for a player.

    Args:
        player_name: Player name
        opponent: Opponent team name
        game_date: YYYY-MM-DD
        risk_mode: "safe", "balanced", or "aggressive"
    """
    opp = resolve_team(opponent) if opponent else ""
    gd = _parse_date(game_date)

    suggestions = suggest_player_props(player_name, opp or "", gd, risk_mode=risk_mode)
    return {
        "player": player_name,
        "opponent": opp,
        "risk_mode": risk_mode,
        "suggestions": suggestions,
        "disclaimer": DISCLAIMER,
    }


# ---------------------------------------------------------------------------
# Tool: make_parlay
# ---------------------------------------------------------------------------

def make_parlay_tool(home_team: str, away_team: str,
                     players: list[str] | None = None,
                     number_of_legs: int = 2,
                     risk_mode: str = "balanced",
                     allow_correlation: bool = False,
                     game_date: str | None = None,
                     constraints: dict | None = None) -> dict:
    """
    Build a parlay for a game.

    Args:
        home_team: Home team
        away_team: Away team
        players: List of player names to include in prop considerations.
                 If None, only moneyline legs are generated.
        number_of_legs: Number of legs (2-6)
        risk_mode: "safe", "balanced", or "aggressive"
        allow_correlation: Allow correlated legs (same game/team)
        game_date: YYYY-MM-DD
        constraints: Additional constraints dict (min_confidence, only_player_props,
                     stat_types, etc.)
    """
    home = resolve_team(home_team)
    away = resolve_team(away_team)
    if not home or not away:
        return {"error": f"Could not resolve teams: {home_team} / {away_team}"}

    gd = _parse_date(game_date)

    # Build all candidate legs
    legs = build_legs_for_game(home, away, gd, players, risk_mode)

    if not legs:
        return {
            "error": "No viable legs found. Provide player names or check team data.",
            "suggestion": "Try: make_parlay with players=['LeBron James', 'Anthony Davis']"
        }

    # Build parlays
    parlays = make_parlay(legs, number_of_legs, risk_mode, allow_correlation, constraints)

    if not parlays:
        return {
            "error": "Could not construct a parlay meeting the criteria. Try fewer legs or lower confidence threshold.",
            "available_legs": [leg_to_dict(l) for l in legs[:5]],
        }

    return {
        "parlays": [parlay_to_dict(p) for p in parlays],
        "total_candidates_evaluated": len(legs),
        "disclaimer": DISCLAIMER,
    }


# ---------------------------------------------------------------------------
# Tool: explain_leg
# ---------------------------------------------------------------------------

def explain_leg(player_name: str | None = None, stat_type: str | None = None,
                line: float | None = None, over_under: str = "over",
                team: str | None = None, opponent: str | None = None,
                game_date: str | None = None,
                description: str | None = None) -> dict:
    """
    Explain why the model likes or dislikes a specific leg.

    Args:
        player_name: For player props
        stat_type: points, rebounds, assists, pra, threes, etc.
        line: The line value
        over_under: over or under
        team: For moneyline legs
        opponent: Opponent team
        game_date: YYYY-MM-DD
        description: Free-text leg description (alternative to structured args)
    """
    gd = _parse_date(game_date)

    if player_name and stat_type:
        opp = resolve_team(opponent) if opponent else ""
        result = predict_player_prop(player_name, stat_type, line, opp or "", gd, over_under=over_under)
        explanation = {
            "leg": f"{player_name} {over_under} {line} {stat_type}" if line else f"{player_name} {stat_type}",
            "projection": result.get("projection"),
            "hit_probability": result.get("hit_probability"),
            "confidence_tier": result.get("confidence_tier"),
            "why_model_likes_it": [r for r in result.get("reasons", []) if "above" not in r.lower() or over_under == "over"],
            "why_model_dislikes_it": result.get("risks", []),
            "data_quality": result.get("data_quality"),
            "suggested_usage": result.get("suggested_usage"),
            "percentiles": result.get("percentiles"),
        }
    elif team:
        home = resolve_team(team)
        opp = resolve_team(opponent) if opponent else ""
        if home and opp:
            ml = predict_moneyline(home, opp, gd)
            explanation = {
                "leg": f"{home} ML",
                "home_win_prob": ml.get("home_win_probability"),
                "away_win_prob": ml.get("away_win_probability"),
                "lean": ml.get("lean"),
                "why": ml.get("reasons", []),
                "caution": ml.get("caution_flags", []),
            }
        else:
            explanation = {"error": "Could not resolve teams"}
    else:
        explanation = {"error": "Provide player_name + stat_type or team + opponent"}

    explanation["disclaimer"] = DISCLAIMER
    return explanation


# ---------------------------------------------------------------------------
# Tool: find_best_legs
# ---------------------------------------------------------------------------

def find_best_legs_tool(home_team: str, away_team: str,
                        players: list[str] | None = None,
                        risk_mode: str = "balanced",
                        stat_types: list[str] | None = None,
                        min_confidence: float = 0.45,
                        game_date: str | None = None) -> dict:
    """
    Find the strongest legs on a slate/game.

    Args:
        home_team: Home team
        away_team: Away team
        players: Player names to evaluate
        risk_mode: safe/balanced/aggressive
        stat_types: Filter to specific stat types
        min_confidence: Minimum confidence score (0-1)
        game_date: YYYY-MM-DD
    """
    home = resolve_team(home_team)
    away = resolve_team(away_team)
    if not home or not away:
        return {"error": f"Could not resolve teams: {home_team} / {away_team}"}

    gd = _parse_date(game_date)
    all_legs = build_legs_for_game(home, away, gd, players, risk_mode)
    best = find_best_legs(all_legs, risk_mode, stat_types, min_confidence)

    return {
        "best_legs": [leg_to_dict(l) for l in best],
        "total_evaluated": len(all_legs),
        "disclaimer": DISCLAIMER,
    }


# ---------------------------------------------------------------------------
# Tool: fade_risky_legs
# ---------------------------------------------------------------------------

def fade_risky_legs_tool(home_team: str, away_team: str,
                         players: list[str] | None = None,
                         game_date: str | None = None) -> dict:
    """
    Find legs to AVOID and explain why.

    Args:
        home_team: Home team
        away_team: Away team
        players: Player names
        game_date: YYYY-MM-DD
    """
    home = resolve_team(home_team)
    away = resolve_team(away_team)
    if not home or not away:
        return {"error": f"Could not resolve teams: {home_team} / {away_team}"}

    gd = _parse_date(game_date)
    all_legs = build_legs_for_game(home, away, gd, players)
    risky = fade_risky_legs(all_legs)

    return {
        "legs_to_avoid": risky,
        "total_evaluated": len(all_legs),
        "disclaimer": DISCLAIMER,
    }
