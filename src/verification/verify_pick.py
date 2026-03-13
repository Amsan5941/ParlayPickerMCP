"""Verification pipeline for player props and moneyline legs."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any

from src.utils.config import ENABLE_LIVE_VERIFICATION, resolve_team, resolve_team_name
from src.utils.logger import get_logger
from src.verification.nba_live import NbaLiveClient, get_live_client

log = get_logger(__name__)


@dataclass(slots=True)
class VerificationResult:
    ok: bool
    corrected_leg: Any | None = None
    warnings: list[str] = field(default_factory=list)
    reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def _verification_time() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _get_field(obj: Any, field: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(field, default)
    return getattr(obj, field, default)


def _set_fields(obj: Any, updates: dict[str, Any]) -> Any:
    cloned = copy.deepcopy(obj)
    if isinstance(cloned, dict):
        cloned.update(updates)
        return cloned
    for key, value in updates.items():
        setattr(cloned, key, value)
    return cloned


def _attach_metadata(leg: Any, metadata: dict[str, Any]) -> Any:
    return _set_fields(leg, {"verification_metadata": metadata})


def _normalize_game_context(game_context: dict[str, Any] | None, leg: Any) -> dict[str, Any]:
    context = dict(game_context or {})
    context.setdefault("home_team", _get_field(leg, "team"))
    context.setdefault("away_team", _get_field(leg, "opponent"))
    game_date = context.get("game_date") or _get_field(leg, "game_date")
    if isinstance(game_date, date):
        game_date = str(game_date)
    context["game_date"] = game_date
    return context


def verify_leg(
    leg: Any,
    game_context: dict[str, Any] | None = None,
    client: NbaLiveClient | None = None,
) -> VerificationResult:
    """Verify a candidate leg against live roster and schedule truth."""
    if not ENABLE_LIVE_VERIFICATION:
        metadata = {
            "verified": False,
            "verified_source": "disabled",
            "verification_time": _verification_time(),
        }
        return VerificationResult(ok=True, corrected_leg=_attach_metadata(leg, metadata), metadata=metadata)

    client = client or get_live_client()
    context = _normalize_game_context(game_context, leg)
    game_date = context.get("game_date")
    leg_type = _get_field(leg, "leg_type") or _get_field(leg, "type")

    metadata = {
        "verified": False,
        "verified_source": "nba_api",
        "verification_time": _verification_time(),
        "game_validated": False,
    }

    if leg_type == "moneyline":
        home = context.get("home_team")
        away = context.get("away_team")
        validation = client.validate_game(home, away, game_date)
        metadata.update({
            "game_validation": validation,
            "game_validated": bool(validation.get("ok") and validation.get("game_exists") in {True, None}),
        })
        if game_date and not validation.get("ok"):
            metadata.update({"reason": validation.get("reason")})
            return VerificationResult(ok=False, reason=validation.get("reason"), metadata=metadata)
        metadata["verified"] = True
        corrected_leg = _attach_metadata(leg, metadata)
        return VerificationResult(ok=True, corrected_leg=corrected_leg, metadata=metadata)

    player_name = _get_field(leg, "player")
    if not player_name:
        metadata.update({"reason": "missing_player_name"})
        return VerificationResult(ok=False, reason="missing_player_name", metadata=metadata)

    current = client.get_current_team(player_name)
    if not current.get("found"):
        metadata.update({
            "reason": "player_not_found",
            "player_name": player_name,
        })
        return VerificationResult(ok=False, reason="player_not_found", metadata=metadata)

    live_team = current.get("current_team_abbrev")
    requested_team = resolve_team(_get_field(leg, "team", "") or "")
    home = resolve_team(context.get("home_team", "") or "")
    away = resolve_team(context.get("away_team", "") or "")
    warnings: list[str] = []

    metadata.update({
        "player_name": current.get("player_name"),
        "current_team": current.get("current_team"),
        "current_team_abbrev": live_team,
    })

    if requested_team and requested_team != live_team and live_team not in {home, away}:
        metadata.update({
            "reason": "team_mismatch",
            "expected_team": resolve_team_name(requested_team),
            "live_team": current.get("current_team"),
        })
        log.warning("Rejected leg for %s due to team mismatch: expected %s, live %s", player_name, requested_team, live_team)
        return VerificationResult(ok=False, reason="team_mismatch", metadata=metadata)

    if home and away and live_team not in {home, away}:
        metadata.update({
            "reason": "player_not_in_game_context",
            "expected_game": f"{home} vs {away}",
        })
        log.warning("Rejected leg for %s because live team %s is not in requested game %s vs %s", player_name, live_team, home, away)
        return VerificationResult(ok=False, reason="player_not_in_game_context", metadata=metadata)

    game_validation = None
    if game_date and home and away:
        game_validation = client.validate_game(home, away, game_date)
        metadata.update({
            "game_validation": game_validation,
            "game_validated": bool(game_validation.get("ok") and game_validation.get("game_exists") in {True, None}),
        })
        if not game_validation.get("ok"):
            metadata.update({"reason": game_validation.get("reason")})
            return VerificationResult(ok=False, reason=game_validation.get("reason"), metadata=metadata)

    corrected_team = live_team
    corrected_opponent = _get_field(leg, "opponent")
    if home and away and live_team in {home, away}:
        corrected_opponent = away if live_team == home else home
        if requested_team and requested_team != live_team:
            warnings.append("corrected_team_from_live_data")

    corrected_leg = _set_fields(leg, {
        "player": current.get("player_name"),
        "team": corrected_team,
        "opponent": corrected_opponent,
    })
    metadata["verified"] = True
    corrected_leg = _attach_metadata(corrected_leg, metadata)
    return VerificationResult(
        ok=True,
        corrected_leg=corrected_leg,
        warnings=warnings,
        metadata=metadata,
    )


def verify_legs(
    legs: list[Any],
    game_context: dict[str, Any] | None = None,
    client: NbaLiveClient | None = None,
) -> tuple[list[Any], list[VerificationResult]]:
    verified: list[Any] = []
    rejected: list[VerificationResult] = []
    client = client or get_live_client()

    for leg in legs:
        result = verify_leg(leg, game_context=game_context, client=client)
        if result.ok:
            verified.append(result.corrected_leg or leg)
        else:
            rejected.append(result)
            log.warning("Rejected stale or invalid leg: %s", result.reason)

    return verified, rejected