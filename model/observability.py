"""
Structured event logging — JSON lines, not free-text prints. Every event a
reviewer would need to reconstruct "what happened" after the fact, from logs
alone, without re-running anything.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone


def emit(event: str, **fields) -> None:
    record = {
        "event": event,
        "ts": datetime.now(timezone.utc).isoformat(),
        **fields,
    }
    print(json.dumps(record), file=sys.stderr)
