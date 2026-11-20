"""A thin wrapper around the google-genai client used by ingestion.

It adds three things every ingestion call needs:
- pacing: at least `seconds_between_calls` between requests (free-tier rate limits, Lesson 5);
- retries with backoff on rate limits, server errors and network errors (but not on a bad
  key or a bad request, which would fail the same way every time);
- a call counter for the ingestion summary.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from google import genai
from google.genai import errors, types

RETRY_WAITS = (5, 15, 30, 60)  # seconds before attempts 2..5
# google-genai 3.x defaults to API "v1", which rejects JSON mode (structured output).
API_VERSION = "v1beta"


class GeminiError(RuntimeError):
    """`fatal` means retrying later in this run is pointless (bad key, bad request)."""

    def __init__(self, message: str, fatal: bool = False):
        super().__init__(message)
        self.fatal = fatal


class Gemini:
    def __init__(self, api_key: str, model: str, seconds_between_calls: float = 4.0,
                 sleep: Callable[[float], None] = time.sleep):
        self.client = genai.Client(api_key=api_key, http_options=types.HttpOptions(api_version=API_VERSION))
        self.model = model
        self.seconds_between_calls = seconds_between_calls
        self.calls = 0
        self._sleep = sleep
        self._last_call = 0.0

    def _pace(self) -> None:
        wait = self._last_call + self.seconds_between_calls - time.monotonic()
        if wait > 0:
            self._sleep(wait)
        self._last_call = time.monotonic()

    def generate(self, contents: Any, config: types.GenerateContentConfig) -> Any:
        """One generate_content call with pacing and retries. Raises GeminiError when it gives up."""
        last: Exception | None = None
        for attempt in range(len(RETRY_WAITS) + 1):
            if attempt:
                self._sleep(RETRY_WAITS[attempt - 1])
            self._pace()
            self.calls += 1
            try:
                return self.client.models.generate_content(model=self.model, contents=contents,
                                                           config=config)
            except errors.ClientError as e:  # 4xx: only "too many requests" is worth retrying
                if e.code != 429:
                    raise GeminiError(f"Gemini rejected the request ({e.code}): {e}", fatal=True) from e
                last = e
            except (errors.ServerError, OSError) as e:  # 5xx, timeouts, dropped connections
                last = e
            except Exception as e:  # httpx transport errors and the like
                if type(e).__module__.split(".")[0] not in {"httpx", "httpcore", "httpx2"}:
                    raise
                last = e
        raise GeminiError(f"Gemini failed {len(RETRY_WAITS) + 1} times: {last}") from last


def finish_reason(response: Any) -> str:
    """'STOP', 'MAX_TOKENS', ... ; 'UNKNOWN' if the response has no candidates."""
    candidates = getattr(response, "candidates", None) or []
    if not candidates or candidates[0].finish_reason is None:
        return "UNKNOWN"
    reason = candidates[0].finish_reason
    return getattr(reason, "name", str(reason))
