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
from src.utils.config import ENABLE_LIVE_VERIFICATION, VERIFY_SCHEDULES, resolve_team, resolve_team_name
from src.utils.logger import get_logger
from src.verification.nba_live import get_live_client
from src.verification.verify_pick import verify_leg, verify_legs

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


def _validate_requested_game(home: str, away: str, gd: date) -> dict | None:
    if not ENABLE_LIVE_VERIFICATION or not VERIFY_SCHEDULES:
        return None
    validation = get_live_client().validate_game(home, away, str(gd))
    if validation.get("ok"):
        return None
    if validation.get("game_exists") is False:
        return {
            "error": f"No live game found for {home} vs {away} on {gd}",
            "verification": validation,
        }
    return {
        "error": "Live schedule verification unavailable",
        "verification": validation,
    }


def _build_prop_leg(result: dict, team: str | None, opponent: str | None, gd: date) -> Leg:
    line_value = result.get("suggested_line", result.get("line"))
    return Leg(
        leg_type="player_prop",
        description=f"{result['player']} {result.get('direction', result.get('over_under', 'over')).title()} {line_value} {result['stat_type']}",
        player=result["player"],
        stat_type=result["stat_type"],
        line=line_value,
        over_under=result.get("direction", result.get("over_under", "over")),
        team=team,
        opponent=opponent,
        game_date=gd,
        hit_probability=result.get("hit_probability", 0.5),
        confidence_tier=result.get("confidence_tier", "Medium"),
        confidence_score=result.get("confidence_score", 0.5),
        reasons=result.get("reasons", []),
        risks=result.get("risks", []),
        projection=result.get("projection"),
        suggested_usage=result.get("suggested_usage", "single only"),
        game_key=f"{team}_{opponent}" if team and opponent else "",
    )


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
    validation_error = _validate_requested_game(home, away, gd)
    if validation_error:
        return validation_error
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
    validation_error = _validate_requested_game(home, away, gd)
    if validation_error:
        return validation_error
    result = predict_moneyline(home, away, gd)
    if ENABLE_LIVE_VERIFICATION:
        verification = get_live_client().validate_game(home, away, str(gd))
        result["verification"] = {
            "verified": verification.get("ok"),
            "verified_source": "nba_api",
            "game_validated": verification.get("game_exists"),
            "details": verification,
        }
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
    client = get_live_client() if ENABLE_LIVE_VERIFICATION else None
    opp_abbrev = resolve_team(opponent) if opponent else ""
    if opponent and not opp_abbrev:
        return {"error": f"Could not resolve opponent: {opponent}"}

    gd = _parse_date(game_date)
    live_info = client.get_current_team(player_name) if client else None
    if client and not live_info.get("found"):
        return {"error": f"Live verification could not resolve player: {player_name}"}

    resolved_player = live_info.get("player_name", player_name) if live_info else player_name
    player_team = live_info.get("current_team_abbrev") if live_info else None

    # Try to determine if home
    is_home = 1  # default
    if opp_abbrev and client and player_team:
        game_validation = client.validate_game(player_team, opp_abbrev, str(gd))
        if not game_validation.get("ok"):
            if game_validation.get("game_exists") is False:
                return {
                    "error": f"No live game found for {resolve_team_name(player_team)} vs {resolve_team_name(opp_abbrev)} on {gd}",
                    "verification": game_validation,
                }
            return {
                "error": "Live verification unavailable for requested player prop",
                "verification": game_validation,
            }
        is_home = 1 if game_validation.get("home_team_abbrev") == player_team else 0
    elif opp_abbrev:
        sched = schedule_on_date(str(gd))
        if not sched.empty:
            resolved_name = find_player_name(player_name)
            if resolved_name:
                rec = player_recent(resolved_name, 1)
                if not rec.empty:
                    historical_team = rec.iloc[0].get("team_abbrev", "")
                    for _, row in sched.iterrows():
                        if historical_team and historical_team in str(row.get("homeTeamId", "")):
                            is_home = 1
                            break
                        elif historical_team and historical_team in str(row.get("awayTeamId", "")):
                            is_home = 0
                            break

    result = predict_player_prop(
        resolved_player, stat_type, line, opp_abbrev or "", gd, is_home, over_under
    )
    if "error" in result:
        return result
    if client and player_team:
        leg = _build_prop_leg(result, player_team, opp_abbrev or None, gd)
        verified = verify_leg(
            leg,
            game_context={
                "home_team": player_team if is_home else opp_abbrev,
                "away_team": opp_abbrev if is_home else player_team,
                "game_date": str(gd),
            } if opp_abbrev else {"game_date": str(gd)},
            client=client,
        )
        if not verified.ok:
            return {"error": f"Live verification rejected leg: {verified.reason}", "verification": verified.metadata}
        result["verification"] = verified.metadata
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
    client = get_live_client() if ENABLE_LIVE_VERIFICATION else None

    resolved_player = player_name
    player_team = None
    is_home = 1
    game_context = {"game_date": str(gd)}
    if client:
        current = client.get_current_team(player_name)
        if not current.get("found"):
            return {"error": f"Live verification could not resolve player: {player_name}"}
        resolved_player = current["player_name"]
        player_team = current.get("current_team_abbrev")
        if opp:
            validation = client.validate_game(player_team, opp, str(gd))
            if not validation.get("ok"):
                if validation.get("game_exists") is False:
                    return {
                        "error": f"No live game found for {resolve_team_name(player_team)} vs {resolve_team_name(opp)} on {gd}",
                        "verification": validation,
                    }
                return {
                    "error": "Live verification unavailable for requested player props",
                    "verification": validation,
                }
            is_home = 1 if validation.get("home_team_abbrev") == player_team else 0
            game_context = {
                "home_team": validation.get("home_team_abbrev"),
                "away_team": validation.get("away_team_abbrev"),
                "game_date": str(gd),
            }

    suggestions = suggest_player_props(resolved_player, opp or "", gd, is_home=is_home, risk_mode=risk_mode)
    verified_suggestions = []
    rejected = []
    for suggestion in suggestions:
        if "error" in suggestion:
            continue
        if client and player_team:
            leg = _build_prop_leg(suggestion, player_team, opp or None, gd)
            verification = verify_leg(leg, game_context=game_context, client=client)
            if not verification.ok:
                rejected.append({
                    "player": suggestion.get("player"),
                    "stat_type": suggestion.get("stat_type"),
                    "reason": verification.reason,
                    "verification": verification.metadata,
                })
                continue
            suggestion["verification"] = verification.metadata
        verified_suggestions.append(suggestion)
    return {
        "player": resolved_player,
        "opponent": opp,
        "risk_mode": risk_mode,
        "suggestions": verified_suggestions,
        "rejected_suggestions": rejected,
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
    validation_error = _validate_requested_game(home, away, gd)
    if validation_error:
        return validation_error

    include_rejected_legs = bool(constraints and constraints.get("include_rejected_legs"))

    # Build all candidate legs
    legs = build_legs_for_game(home, away, gd, players, risk_mode)
    verified_legs, rejected_legs = verify_legs(
        legs,
        game_context={"home_team": home, "away_team": away, "game_date": str(gd)},
        client=get_live_client() if ENABLE_LIVE_VERIFICATION else None,
    ) if ENABLE_LIVE_VERIFICATION else (legs, [])
    legs = verified_legs

    if not legs:
        response = {
            "error": "No viable legs found. Provide player names or check team data.",
            "suggestion": "Try: make_parlay with players=['LeBron James', 'Anthony Davis']",
        }
        if include_rejected_legs:
            response["rejected_legs"] = [r.metadata for r in rejected_legs]
        return response

    # Build parlays
    parlays = make_parlay(legs, number_of_legs, risk_mode, allow_correlation, constraints)

    if not parlays:
        return {
            "error": "Could not construct a parlay meeting the criteria. Try fewer legs or lower confidence threshold.",
            "available_legs": [leg_to_dict(l) for l in legs[:5]],
        }

    response = {
        "parlays": [parlay_to_dict(p) for p in parlays],
        "total_candidates_evaluated": len(legs),
        "disclaimer": DISCLAIMER,
    }
    if include_rejected_legs:
        response["rejected_legs"] = [r.metadata for r in rejected_legs]
    return response


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
    validation_error = _validate_requested_game(home, away, gd)
    if validation_error:
        return validation_error
    all_legs = build_legs_for_game(home, away, gd, players, risk_mode)
    rejected_legs = []
    if ENABLE_LIVE_VERIFICATION:
        all_legs, rejected_legs = verify_legs(
            all_legs,
            game_context={"home_team": home, "away_team": away, "game_date": str(gd)},
            client=get_live_client(),
        )
    best = find_best_legs(all_legs, risk_mode, stat_types, min_confidence)

    return {
        "best_legs": [leg_to_dict(l) for l in best],
        "total_evaluated": len(all_legs),
        "rejected_legs": [r.metadata for r in rejected_legs],
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
    validation_error = _validate_requested_game(home, away, gd)
    if validation_error:
        return validation_error
    all_legs = build_legs_for_game(home, away, gd, players)
    rejected_legs = []
    if ENABLE_LIVE_VERIFICATION:
        all_legs, rejected_legs = verify_legs(
            all_legs,
            game_context={"home_team": home, "away_team": away, "game_date": str(gd)},
            client=get_live_client(),
        )
    risky = fade_risky_legs(all_legs)

    return {
        "legs_to_avoid": risky,
        "total_evaluated": len(all_legs),
        "rejected_legs": [r.metadata for r in rejected_legs],
        "disclaimer": DISCLAIMER,
    }
