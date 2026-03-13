"""JSON-backed local cache for live NBA verification data."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from src.utils.config import LIVE_CACHE_DIR, LIVE_CACHE_TTL_HOURS
from src.utils.logger import get_logger

log = get_logger(__name__)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def isoformat_utc(value: datetime | None = None) -> str:
    value = value or utc_now()
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_utc(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None


@dataclass(slots=True)
class CacheResult:
    payload: Any
    fetched_at: str
    stale: bool
    refreshed: bool
    source: str
    warning: str | None = None


class LiveCache:
    """Simple TTL cache persisted to JSON files."""

    def __init__(self, cache_dir: Path | None = None, ttl_hours: int | None = None):
        self.cache_dir = Path(cache_dir or LIVE_CACHE_DIR)
        self.ttl_hours = ttl_hours or LIVE_CACHE_TTL_HOURS
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def load(self, key: str) -> dict[str, Any] | None:
        path = self._path(key)
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def save(self, key: str, payload: Any, ttl_hours: int | None = None) -> dict[str, Any]:
        envelope = {
            "key": key,
            "fetched_at": isoformat_utc(),
            "ttl_hours": ttl_hours or self.ttl_hours,
            "payload": payload,
        }
        path = self._path(key)
        tmp_path = path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(envelope, indent=2, sort_keys=True))
        tmp_path.replace(path)
        return envelope

    def is_stale(self, envelope: dict[str, Any], ttl_hours: int | None = None) -> bool:
        fetched_at = parse_utc(envelope.get("fetched_at"))
        if fetched_at is None:
            return True
        ttl = ttl_hours if ttl_hours is not None else int(envelope.get("ttl_hours", self.ttl_hours))
        return utc_now() - fetched_at > timedelta(hours=ttl)

    def get_or_refresh(
        self,
        key: str,
        loader: Callable[[], Any],
        ttl_hours: int | None = None,
        force_refresh: bool = False,
        allow_stale_on_error: bool = True,
    ) -> CacheResult:
        envelope = self.load(key)
        stale = envelope is None or self.is_stale(envelope, ttl_hours)

        if envelope and not stale and not force_refresh:
            log.info("Using live cache for %s", key)
            return CacheResult(
                payload=envelope["payload"],
                fetched_at=envelope["fetched_at"],
                stale=False,
                refreshed=False,
                source="cache",
            )

        try:
            payload = loader()
            envelope = self.save(key, payload, ttl_hours)
            log.info("Refreshed live cache for %s", key)
            return CacheResult(
                payload=payload,
                fetched_at=envelope["fetched_at"],
                stale=False,
                refreshed=True,
                source="api",
            )
        except Exception as exc:  # pragma: no cover - exercised via tests with mocks
            if envelope and allow_stale_on_error:
                log.warning("Live cache refresh failed for %s; using stale cache: %s", key, exc)
                return CacheResult(
                    payload=envelope["payload"],
                    fetched_at=envelope["fetched_at"],
                    stale=True,
                    refreshed=False,
                    source="stale-cache",
                    warning=str(exc),
                )
            raise