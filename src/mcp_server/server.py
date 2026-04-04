"""
NBA Parlay Maker — MCP Server.

Exposes NBA betting analytics tools via the Model Context Protocol
for use with Claude Desktop or any MCP-compatible client.

Run with:
    python -m src.mcp_server.server
"""

import json
import sys
from mcp.server.fastmcp import FastMCP

from src.tools.betting_tools import (
    analyze_game,
    evaluate_moneyline,
    evaluate_player_prop,
    suggest_player_props_tool,
    make_parlay_tool,
    explain_leg,
    find_best_legs_tool,
    fade_risky_legs_tool,
)
from src.utils.logger import get_logger

log = get_logger(__name__)

# Create the MCP server
mcp = FastMCP("NBA Parlay Maker")


# ---------------------------------------------------------------------------
# Tool registrations
# ---------------------------------------------------------------------------

@mcp.tool()
def analyze_game_tool(home_team: str, away_team: str,
                      game_date: str = "") -> str:
    """
    Analyze a game matchup between two NBA teams.

    Provides: projected winner, win probabilities, confidence tier, key reasons,
    caution flags, and recent form.

    Examples:
      - analyze_game_tool("Lakers", "Celtics")
      - analyze_game_tool("LAL", "BOS", "2026-03-15")

    Args:
        home_team: Home team name (city, abbreviation, or full name)
        away_team: Away team name
        game_date: Optional date in YYYY-MM-DD format (defaults to today)
    """
    result = analyze_game(home_team, away_team, game_date or None)
    return json.dumps(result, indent=2, default=str)


@mcp.tool()
def evaluate_moneyline_tool(home_team: str, away_team: str,
                            game_date: str = "") -> str:
    """
    Evaluate a moneyline bet for an NBA game.

    Returns win probabilities, lean (who to pick), projected margin,
    explanation, and caution flags.

    Examples:
      - evaluate_moneyline_tool("Nuggets", "Suns")
      - evaluate_moneyline_tool("DEN", "PHX", "2026-03-15")

    Args:
        home_team: Home team name
        away_team: Away team name
        game_date: Optional YYYY-MM-DD
    """
    result = evaluate_moneyline(home_team, away_team, game_date or None)
    return json.dumps(result, indent=2, default=str)


@mcp.tool()
def evaluate_player_prop_tool(player_name: str, stat_type: str,
                              line: float = 0.0, opponent: str = "",
                              game_date: str = "",
                              over_under: str = "over") -> str:
    """
    Evaluate a player prop bet.

    Returns: projection, projection_std, hit probability, confidence tier,
    value_score (edge-weighted), reasons, risks, percentiles, and suggested usage.

    Stat types: points, rebounds, assists, pra, pa (pts+ast), ra (reb+ast),
                threes, steals, blocks

    Examples:
      - evaluate_player_prop_tool("LeBron James", "points", 27.5, "Celtics")
      - evaluate_player_prop_tool("Jokic", "pra", 46.5, "Lakers", "2026-03-15")
      - evaluate_player_prop_tool("Tatum", "pa", 38.5, "Heat", over_under="over")
      - evaluate_player_prop_tool("Curry", "threes", 4.5, "Lakers")

    Args:
        player_name: Full or partial player name
        stat_type: points, rebounds, assists, pra, pa, ra, threes, steals, blocks
        line: Betting line value (0 = no line, will return projection + suggested lines)
        opponent: Opponent team name
        game_date: Optional YYYY-MM-DD
        over_under: "over" or "under"
    """
    actual_line = line if line > 0 else None
    result = evaluate_player_prop(
        player_name, stat_type, actual_line,
        opponent or None, game_date or None, over_under
    )
    return json.dumps(result, indent=2, default=str)


@mcp.tool()
def suggest_props_tool(player_name: str, opponent: str = "",
                       game_date: str = "",
                       risk_mode: str = "balanced") -> str:
    """
    Suggest the best prop bets for a player in their next game.

    Returns ranked prop ideas across ALL stat categories — points, rebounds,
    assists, PRA, PA (pts+ast), RA (reb+ast), threes, steals, blocks — with
    hit probabilities, value_score (edge-weighted), and confidence tiers.

    Lines are snapped to realistic 0.5 sportsbook increments.

    Risk modes:
      - safe: conservative lines (~12% below projection), highest hit probability
      - balanced: lines ~6% below projection, good balance of value and safety
      - aggressive: lines near/above projection, higher upside

    Examples:
      - suggest_props_tool("Anthony Davis", "Warriors")
      - suggest_props_tool("Tatum", "Heat", risk_mode="safe")
      - suggest_props_tool("Curry", "Suns", risk_mode="balanced")

    Args:
        player_name: Player name
        opponent: Opponent team name
        game_date: Optional YYYY-MM-DD
        risk_mode: "safe", "balanced", or "aggressive"
    """
    result = suggest_player_props_tool(
        player_name, opponent or None, game_date or None, risk_mode
    )
    return json.dumps(result, indent=2, default=str)


@mcp.tool()
def make_parlay_builder(home_team: str, away_team: str,
                        players: str = "",
                        number_of_legs: int = 2,
                        risk_mode: str = "balanced",
                        allow_correlation: bool = False,
                        game_date: str = "",
                        only_player_props: bool = False,
                        min_confidence: float = 0.0,
                        include_rejected_legs: bool = False) -> str:
    """
    Build a parlay for an NBA game.

    Generates ranked parlays with per-leg confidence, edge-weighted value_score,
    correlation warnings, combined hit probability, and risk explanations.

    Players are AUTO-DISCOVERED from the live roster when not provided — just
    pass the two teams and the system finds the top starters automatically.

    Legs are ranked by value_score (hit probability + edge cushion) and filtered
    for minimum edge (projection must meaningfully exceed the line). Parlays are
    diversified across stat types for better independence.

    Covers all prop types: points, rebounds, assists, PRA, PA, RA, threes, steals, blocks.

    Risk modes:
      - safe: High-confidence, min 6% edge, max 2 legs
      - balanced: min 3% edge, max 3 legs
      - aggressive: No edge filter, up to 6 legs

    Examples:
      - make_parlay_builder("Lakers", "Celtics")  ← auto-discovers players!
      - make_parlay_builder("Lakers", "Celtics", "LeBron James,Anthony Davis,Jayson Tatum", 2, "safe")
      - make_parlay_builder("Knicks", "Heat", number_of_legs=3, risk_mode="balanced")

    Args:
        home_team: Home team
        away_team: Away team
        players: Comma-separated player names (leave empty to auto-discover)
        number_of_legs: 2-6
        risk_mode: safe/balanced/aggressive
        allow_correlation: Allow correlated legs in same game
        game_date: Optional YYYY-MM-DD
        only_player_props: Only use player prop legs (no moneyline)
        min_confidence: Minimum confidence score for legs (0-1)
        include_rejected_legs: Include live-verification rejected legs in response
    """
    player_list = [p.strip() for p in players.split(",") if p.strip()] if players else None
    constraints = {}
    if only_player_props:
        constraints["only_player_props"] = True
    if min_confidence > 0:
        constraints["min_confidence"] = min_confidence
    if include_rejected_legs:
        constraints["include_rejected_legs"] = True

    result = make_parlay_tool(
        home_team, away_team, player_list, number_of_legs,
        risk_mode, allow_correlation, game_date or None, constraints or None
    )
    return json.dumps(result, indent=2, default=str)


@mcp.tool()
def explain_leg_tool(player_name: str = "", stat_type: str = "",
                     line: float = 0.0, over_under: str = "over",
                     team: str = "", opponent: str = "",
                     game_date: str = "") -> str:
    """
    Explain WHY the model likes or dislikes a specific leg.

    Provide either player prop details OR moneyline details.

    Examples:
      - explain_leg_tool(player_name="Tatum", stat_type="points", line=29.5, opponent="Heat")
      - explain_leg_tool(team="Lakers", opponent="Celtics")

    Args:
        player_name: For player props
        stat_type: points, rebounds, assists, pra, threes
        line: Line value (0 if none)
        over_under: over or under
        team: For moneyline legs
        opponent: Opponent team
        game_date: Optional YYYY-MM-DD
    """
    result = explain_leg(
        player_name or None, stat_type or None,
        line if line > 0 else None, over_under,
        team or None, opponent or None, game_date or None
    )
    return json.dumps(result, indent=2, default=str)


@mcp.tool()
def find_best_legs_on_slate(home_team: str, away_team: str,
                            players: str = "",
                            risk_mode: str = "balanced",
                            stat_types: str = "",
                            min_confidence: float = 0.45,
                            game_date: str = "") -> str:
    """
    Find the strongest legs for a game, ranked by value_score (edge + hit probability).

    Players are AUTO-DISCOVERED when not provided. Covers all prop types:
    points, rebounds, assists, PRA, PA (pts+ast), RA (reb+ast), threes, steals, blocks.

    Examples:
      - find_best_legs_on_slate("Lakers", "Celtics")  ← auto-discovers players!
      - find_best_legs_on_slate("Lakers", "Celtics", "LeBron,Tatum,Brown,AD")
      - find_best_legs_on_slate("Knicks", "Heat", stat_types="points,pra,pa")

    Args:
        home_team: Home team
        away_team: Away team
        players: Comma-separated player names (leave empty to auto-discover)
        risk_mode: safe/balanced/aggressive
        stat_types: Comma-separated stat types to filter (empty = all)
        min_confidence: Minimum confidence threshold
        game_date: Optional YYYY-MM-DD
    """
    player_list = [p.strip() for p in players.split(",") if p.strip()] if players else None
    st_list = [s.strip() for s in stat_types.split(",") if s.strip()] if stat_types else None
    result = find_best_legs_tool(
        home_team, away_team, player_list, risk_mode,
        st_list, min_confidence, game_date or None
    )
    return json.dumps(result, indent=2, default=str)


@mcp.tool()
def fade_risky_legs_on_slate(home_team: str, away_team: str,
                             players: str = "",
                             game_date: str = "") -> str:
    """
    Find legs to AVOID on a slate with explanations.

    Returns risky legs with risk scores and reasons to fade them.

    Examples:
      - fade_risky_legs_on_slate("Lakers", "Celtics", "LeBron,Tatum,Brown")

    Args:
        home_team: Home team
        away_team: Away team
        players: Comma-separated player names
        game_date: Optional YYYY-MM-DD
    """
    player_list = [p.strip() for p in players.split(",") if p.strip()] if players else None
    result = fade_risky_legs_tool(
        home_team, away_team, player_list, game_date or None
    )
    return json.dumps(result, indent=2, default=str)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    """Run the MCP server."""
    log.info("Starting NBA Parlay Maker MCP server...")
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
