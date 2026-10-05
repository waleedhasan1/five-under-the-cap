"""Disk cache with rate limiting and retry for raw API responses."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


def fetch_cached(
    cache_file: Path,
    fetch: Callable[[], dict[str, Any]],
    *,
    sleep_seconds: float = 1.0,
    max_retries: int = 4,
    backoff_factor: float = 2.0,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Return the JSON payload at ``cache_file``, calling ``fetch`` only if it is absent.

    A fresh response is written to disk before it is returned, so a rerun never
    re-hits the network for the same request.

    Args:
        cache_file: Where the raw response lives.
        fetch: Zero-argument callable that performs the request and returns parsed JSON.
        sleep_seconds: Pause before every network call (rate limit).
        max_retries: Attempts after the first failure before giving up.
        backoff_factor: Multiplier on the pause after each failure.
        sleep: Injected for tests.

    Raises:
        The last exception from ``fetch`` once retries are exhausted.
    """
    if cache_file.exists():
        with open(cache_file) as f:
            cached: dict[str, Any] = json.load(f)
        return cached

    delay = sleep_seconds
    for attempt in range(max_retries + 1):
        sleep(delay)
        try:
            payload = fetch()
            break
        except Exception as exc:
            if attempt == max_retries:
                raise
            log.warning("Request for %s failed (%s); retry %d", cache_file.name, exc, attempt + 1)
            delay *= backoff_factor

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_file.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f)
    tmp.replace(cache_file)
    return payload
