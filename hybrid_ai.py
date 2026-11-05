"""GrowthLab AI adapter: local Ollama or Google Gemini cloud inference.

Uses Gemini's OpenAI-compatible chat completions endpoint to preserve the
existing main.py chat() contract for streaming and non-streaming responses.
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
    """Gemini rejected a request or the API is temporarily cooling down."""


class HybridAIClient:
    _lock = threading.Lock()
    _cooldown_until = 0.0
    _endpoint = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"

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
        payload = {
            "model": os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
            "messages": messages or [],
            "stream": bool(stream),
            "temperature": options.get("temperature", 0.2),
            "max_tokens": options.get("num_predict", 1200),
        }
        response = requests.post(
            self._endpoint,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            stream=stream,
            timeout=(15, 120),
        )
        if response.status_code == 429:
            seconds = self._retry_seconds(response)
            with self._lock:
                self.__class__._cooldown_until = max(
                    self.__class__._cooldown_until, time.monotonic() + seconds
                )
            response.close()
            logging.warning("Gemini returned HTTP 429; cooldown %.1f seconds", seconds)
            raise GeminiRateLimitError(
                f"Gemini rate limit reached; retry after approximately {seconds:.0f} seconds."
            )
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            status = response.status_code
            logging.error("Gemini API returned HTTP %s", status)
            response.close()
            raise RuntimeError(f"Gemini API request failed (HTTP {status}).") from exc
        except Exception:
            response.close()
            raise

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
