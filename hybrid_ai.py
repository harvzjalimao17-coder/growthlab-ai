"""Local Ollama / Groq adapter with bounded Groq rate-limit protection."""
import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from types import SimpleNamespace

import requests


class GroqRateLimitError(RuntimeError):
    """Groq rejected a request or is in a temporary cooldown."""


class HybridAIClient:
    _lock = threading.Lock()
    _cooldown_until = 0.0

    def __init__(self, local_client):
        self.local_client = local_client

    @staticmethod
    def _retry_seconds(response):
        """Honor Retry-After / Groq reset hints; bound waits to 5 minutes."""
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
        if seconds is None:
            # Groq may provide a relative reset duration, e.g. "1m23s".
            reset = response.headers.get("x-ratelimit-reset-tokens", "")
            import re
            match = re.fullmatch(r"(?:(\d+)m)?(?:(\d+(?:\.\d+)?)s)?", reset.strip())
            if match and any(match.groups()):
                seconds = int(match.group(1) or 0) * 60 + float(match.group(2) or 0)
        return max(2.0, min(300.0, seconds if seconds is not None else 30.0))

    def chat(self, model=None, messages=None, options=None, stream=False, **kwargs):
        if os.getenv("GROWTHLAB_CLOUD_MODE", "").lower() != "true":
            return self.local_client.chat(
                model=model, messages=messages, options=options,
                stream=stream, **kwargs
            )

        api_key = os.getenv("GROQ_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is missing")

        with self._lock:
            remaining = self._cooldown_until - time.monotonic()
        if remaining > 0:
            raise GroqRateLimitError(
                f"Groq is rate-limited; retry in approximately {remaining:.0f} seconds."
            )

        payload = {
            "model": os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            "messages": messages or [],
            "stream": stream,
            "temperature": (options or {}).get("temperature", 0.2),
            "max_tokens": (options or {}).get("num_predict", 1200),
        }
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}",
                     "Content-Type": "application/json"},
            json=payload, stream=stream, timeout=120,
        )
        if response.status_code == 429:
            seconds = self._retry_seconds(response)
            with self._lock:
                self.__class__._cooldown_until = max(
                    self._cooldown_until, time.monotonic() + seconds
                )
            response.close()
            logging.warning("Groq returned HTTP 429; cooldown %.1f seconds", seconds)
            raise GroqRateLimitError(
                f"Groq rate limit reached; retry after approximately {seconds:.0f} seconds."
            )
        try:
            response.raise_for_status()
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
                    if not line or not line.startswith(b"data: "):
                        continue
                    raw = line[6:].decode("utf-8")
                    if raw == "[DONE]":
                        break
                    data = json.loads(raw)
                    content = data["choices"][0].get("delta", {}).get("content") or ""
                    if content:
                        yield SimpleNamespace(message=SimpleNamespace(content=content))
            finally:
                response.close()

        return generate()

    def list(self):
        return self.local_client.list()
