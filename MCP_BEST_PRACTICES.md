# NBA Parlay MCP - Best Use Instructions

Use this file as a quick operating guide for getting the highest-quality results from this MCP.

## 1) Copy-Paste Assistant Instruction Block

Paste this into your chat as your first message before asking for picks:

```text
You are helping me use my NBA Parlay MCP.

Always follow this process:
1. Resolve teams and players to current NBA entities.
2. Use explicit game dates in YYYY-MM-DD (never assume "today" without confirming date).
3. Prefer live-verified results. If verification fails, show the reason clearly.
4. Reject or avoid legs with inactive players or invalid game context.
5. For parlays, include confidence, correlation warnings, and rejected legs when possible.
6. If data is stale or unavailable, say it directly and suggest a safer fallback.

When building parlays:
- Default to risk_mode="safe" unless I request otherwise.
- Default to number_of_legs=2 unless I request otherwise.
- Prefer min_confidence >= 0.55 for safer builds.
- Include rejected legs so I can see what was filtered out.

Output format:
- Short summary first.
- Then final recommended legs.
- Then "Reasons", "Risks", and "Verification status".
```

## 2) Recommended Workflow

Run in this order for best results:

1. `analyze_game_tool` for game-level context.
2. `find_best_legs_on_slate` to shortlist candidates.
3. `evaluate_player_prop_tool` on 2-5 finalists.
4. `make_parlay_builder` with:
   - `include_rejected_legs=true`
   - `risk_mode="safe"` or `"balanced"`
   - `min_confidence` set (0.55 safe, 0.45 balanced)
5. `fade_risky_legs_on_slate` as a final filter.

## 3) High-Quality Prompt Templates

Use exact dates, exact teams, and explicit players.

```text
Analyze Knicks vs Heat for 2026-03-14, then give me the safest moneyline angle.
```

```text
Find best legs for Lakers vs Celtics on 2026-03-14 using LeBron James, Anthony Davis, Jayson Tatum, Jaylen Brown. Keep only confidence >= 0.55.
```

```text
Evaluate Jalen Brunson over 26.5 points vs Heat on 2026-03-14 and explain both reasons and risks.
```

```text
Build a safe 2-leg parlay for Nuggets vs Suns on 2026-03-14 with Jokic, Murray, Booker, Durant.
Use min_confidence 0.55, avoid correlated legs if possible, and include rejected legs.
```

```text
Fade risky legs for Warriors vs Timberwolves on 2026-03-14 with Curry, Edwards, Gobert, Draymond.
```

## 4) Guardrails That Improve Results

- Always provide the game date as `YYYY-MM-DD`.
- Always pass full player names when possible.
- Keep first pass to 4-8 players max for cleaner filtering.
- Start with 2-leg parlays; only increase legs after reviewing risks.
- If same-game parlay, pay extra attention to correlation warnings.

## 5) Troubleshooting Quick Guide

- `No live game found`: wrong date or wrong matchup.
- `player_not_in_game_context`: player team is not in that game.
- `player_inactive`: player is listed inactive; remove the leg.
- `team_mismatch`: player/team input is outdated; trust live correction.
- `Live verification unavailable`: do not force an aggressive parlay; reduce stake or skip.

## 6) Suggested Daily Routine

1. Validate slate with explicit date.
2. Build shortlist of best legs.
3. Deep-check top props.
4. Build one safe parlay and one balanced parlay.
5. Keep only legs with clear verification and understandable risk.
