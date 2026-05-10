"""
http_retry.py — Exponential-backoff HTTP GET utility for BetIQ.

Usage:
    from http_retry import get_with_retry

    r = get_with_retry(url, params=params, timeout=15)
    if r is not None:
        data = r.json()

Retry policy:
  • Up to max_attempts total tries (default 3)
  • Waits: 2s → 4s → 8s  (doubles each attempt)
  • Retries on: connection errors, timeouts, 429, 5xx
  • Does NOT retry on: 4xx (except 429), bad API keys, etc.
  • Returns None after all attempts exhausted (never raises)
"""

from __future__ import annotations

import logging
import time
from typing import Any

import requests

log = logging.getLogger(__name__)

_RETRY_DELAYS = [2, 4, 8]   # seconds between attempts 1→2, 2→3, 3→4


def get_with_retry(
    url: str,
    *,
    params:      dict | None = None,
    headers:     dict | None = None,
    timeout:     int         = 15,
    max_attempts: int        = 3,
    label:       str         = "",
) -> requests.Response | None:
    """
    GET url with exponential backoff.

    Parameters
    ----------
    url          : Full URL to fetch.
    params       : Query parameters dict (passed to requests).
    headers      : HTTP headers dict (passed to requests).
    timeout      : Per-attempt socket timeout in seconds.
    max_attempts : Total number of tries before giving up.
    label        : Short string for log messages (e.g. "OddsAPI" or "ESPN").

    Returns
    -------
    requests.Response on success (status < 400, excluding handled retries).
    None if all attempts failed.
    """
    tag = f"[{label}] " if label else ""
    delays = _RETRY_DELAYS[:max_attempts - 1]   # one fewer delay than attempts

    for attempt in range(1, max_attempts + 1):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=timeout)

            # 429 — rate limited: back off regardless of attempt number
            if r.status_code == 429:
                wait = delays[attempt - 1] if attempt - 1 < len(delays) else delays[-1]
                log.warning(f"{tag}429 rate-limited (attempt {attempt}/{max_attempts}). Waiting {wait}s.")
                time.sleep(wait)
                continue

            # 5xx — server error: worth retrying
            if r.status_code >= 500:
                wait = delays[attempt - 1] if attempt - 1 < len(delays) else delays[-1]
                log.warning(
                    f"{tag}HTTP {r.status_code} on attempt {attempt}/{max_attempts}. "
                    f"Waiting {wait}s before retry."
                )
                if attempt < max_attempts:
                    time.sleep(wait)
                continue

            # 4xx (except 429) — client error: don't retry
            if r.status_code >= 400:
                log.error(f"{tag}HTTP {r.status_code} — not retrying: {url}")
                return None

            return r  # success

        except requests.exceptions.Timeout:
            wait = delays[attempt - 1] if attempt - 1 < len(delays) else delays[-1]
            log.warning(f"{tag}Timeout on attempt {attempt}/{max_attempts}. Waiting {wait}s.")
            if attempt < max_attempts:
                time.sleep(wait)

        except requests.exceptions.ConnectionError as exc:
            wait = delays[attempt - 1] if attempt - 1 < len(delays) else delays[-1]
            log.warning(f"{tag}Connection error on attempt {attempt}/{max_attempts}: {exc}. Waiting {wait}s.")
            if attempt < max_attempts:
                time.sleep(wait)

        except requests.exceptions.RequestException as exc:
            log.error(f"{tag}Unrecoverable request error: {exc}")
            return None

    log.error(f"{tag}All {max_attempts} attempts failed for {url}")
    return None
