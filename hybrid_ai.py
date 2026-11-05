"""GrowthLab AI adapter: local Ollama or Google Gemini cloud inference.

Preserves the Ollama-style chat() response contract for streaming and
non-streaming callers in main.py. Groq is not used.
"""
import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from types import SimpleNamespace

import requests


class GeminiRateLimitError(RuntimeError):
    """Gemini has limited requests and requires a cooldown."""


class HybridAIClient:
    _lock = threading.Lock()
    _cooldown_until = 0.0
    _endpoint = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
    _transient_statuses = frozenset((500, 502, 503, 504))
    _max_attempts = 3

    def __init__(self, local_client):
        self.local_client = local_client

    @staticmethod
    def _retry_seconds(response):
        header = response.headers.get("Retry-After", "").strip()
        seconds = None
        if header:
            try:
                seconds = float(header)
            except ValueError:
                try:
                    dt = parsedate_to_datetime(header)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    seconds = (dt - datetime.now(timezone.utc)).total_seconds()
                except (TypeError, ValueError, OverflowError):
                    pass
        return max(2.0, min(300.0, seconds if seconds is not None else 30.0))

    @staticmethod
    def _error_summary(response):
        """Extract a short provider error without logging keys or request data."""
        try:
            error = response.json().get("error", {})
            if isinstance(error, dict):
                code = str(error.get("status") or "")[:48]
                message = str(error.get("message") or "")
                # Avoid printing arbitrary provider text that might echo credentials.
                message = message.replace(os.getenv("GEMINI_API_KEY", "\x00"), "[REDACTED]")
                return (code + ": " + message[:240]).strip(": ")
        except (ValueError, TypeError, AttributeError):
            pass
        return "No structured provider error returned"

    def chat(self, model=None, messages=None, options=None, stream=False, **kwargs):
        if os.getenv("GROWTHLAB_CLOUD_MODE", "").lower() != "true":
            return self.local_client.chat(
                model=model, messages=messages, options=options,
                stream=stream, **kwargs
            )

        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is missing from the environment")

        with self._lock:
            remaining = self.__class__._cooldown_until - time.monotonic()
        if remaining > 0:
            raise GeminiRateLimitError(
                f"Gemini is rate-limited; retry in approximately {max(1, remaining):.0f} seconds."
            )

        options = options or {}
        model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip() or "gemini-2.5-flash"
        payload = {
            "model": model_name,
            "messages": messages or [],
            "stream": bool(stream),
            "temperature": options.get("temperature", 0.2),
            "max_tokens": options.get("num_predict", 1200),
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        response = None
        for attempt in range(1, self._max_attempts + 1):
            try:
                response = requests.post(
                    self._endpoint,
                    headers=headers,
                    json=payload,
                    stream=stream,
                    timeout=(15, 120),
                )
            except (requests.ConnectionError, requests.Timeout) as exc:
                logging.warning(
                    "Gemini connection failure: model=%s attempt=%s/%s error=%s",
                    model_name, attempt, self._max_attempts, type(exc).__name__,
                )
                if attempt == self._max_attempts:
                    raise RuntimeError("Gemini connection failed after retries.") from exc
                time.sleep(min(2 ** attempt, 8))
                continue

            status = response.status_code
            if status == 429:
                seconds = self._retry_seconds(response)
                with self._lock:
                    self.__class__._cooldown_until = max(
                        self.__class__._cooldown_until, time.monotonic() + seconds
                    )
                response.close()
                logging.warning("Gemini HTTP 429: model=%s cooldown=%.1fs", model_name, seconds)
                raise GeminiRateLimitError(
                    f"Gemini rate limit reached; retry after approximately {seconds:.0f} seconds."
                )

            if status in self._transient_statuses:
                logging.warning(
                    "Gemini temporary HTTP %s: model=%s attempt=%s/%s detail=%s",
                    status, model_name, attempt, self._max_attempts,
                    self._error_summary(response),
                )
                response.close()
                if attempt < self._max_attempts:
                    time.sleep(min(2 ** attempt, 8))
                    continue
                raise RuntimeError(
                    f"Gemini API temporarily unavailable (HTTP {status}) after {attempt} attempts."
                )

            if not response.ok:
                logging.error(
                    "Gemini API HTTP %s: model=%s detail=%s",
                    status, model_name, self._error_summary(response),
                )
                response.close()
                raise RuntimeError(f"Gemini API request failed (HTTP {status}).")
            break

        if not stream:
            try:
                data = response.json()
                return SimpleNamespace(message=SimpleNamespace(
                    content=data["choices"][0]["message"].get("content") or ""
                ))
            finally:
                response.close()

        def generate():
            try:
                for line in response.iter_lines():
                    if not line:
                        continue
                    if isinstance(line, bytes):
                        line = line.decode("utf-8", errors="replace")
                    if not line.startswith("data:"):
                        continue
                    raw = line[5:].strip()
                    if raw == "[DONE]":
                        break
                    if not raw:
                        continue
                    data = json.loads(raw)
                    if "error" in data:
                        logging.error("Gemini streaming response reported an error")
                        raise RuntimeError("Gemini streaming response returned an error")
                    for choice in data.get("choices", []):
                        content = (choice.get("delta") or {}).get("content") or ""
                        if content:
                            yield SimpleNamespace(message=SimpleNamespace(content=content))
            finally:
                response.close()

        return generate()

    def list(self):
        return self.local_client.list()
