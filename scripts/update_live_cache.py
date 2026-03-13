"""Manual refresh utility for live NBA verification caches."""

from __future__ import annotations

import argparse
import json

from src.verification.nba_live import get_live_client


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh live nba_api-backed caches")
    parser.add_argument(
        "--date",
        dest="dates",
        action="append",
        default=[],
        help="Optional game date to refresh schedule cache for (YYYY-MM-DD). Can be passed multiple times.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force refresh even if cache entries are still within TTL.",
    )
    args = parser.parse_args()

    summary = get_live_client().refresh_all(game_dates=args.dates, force=args.force)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()