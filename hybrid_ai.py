"""GrowthLab AI: Ollama-compatible adapter for Google's native Gemini API.

Uses generateContent / streamGenerateContent, not the experimental Interactions API.
No additional SDK dependency. Never logs API keys or prompt contents.
"""
import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from types import SimpleNamespace
from urllib.parse import quote

import requests


class GeminiRateLimitError(RuntimeError):
    """Gemini is rate-limited or in a provider cooldown."""


class HybridAIClient:
    _lock = threading.Lock()
    _cooldown_until = 0.0
    _base_url = "https://generativelanguage.googleapis.com/v1beta/models"
    _transient_statuses = {500, 502, 503, 504}

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
    def _error_detail(response):
        try:
            error = response.json().get("error", {})
            if isinstance(error, dict):
                return str(error.get("message") or error.get("status") or "Provider did not include an error message")[:240]
        except (ValueError, AttributeError, TypeError):
            pass
        return "Provider did not include an error message"

    @staticmethod
    def _request_payload(messages, options):
        system_parts, contents = [], []
        for message in messages or []:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role") or "user").lower()
            value = message.get("content") or ""
            if isinstance(value, list):
                value = "\n".join(str(p.get("text", "")) if isinstance(p, dict) else str(p) for p in value)
            value = str(value).strip()
            if not value:
                continue
            if role == "system":
                system_parts.append(value)
            elif role in ("user", "assistant"):
                gemini_role = "model" if role == "assistant" else "user"
                if contents and contents[-1]["role"] == gemini_role:
                    contents[-1]["parts"].append({"text": value})
                else:
                    contents.append({"role": gemini_role, "parts": [{"text": value}]})
        if not contents:
            raise ValueError("Gemini requires at least one conversational message")
        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": options.get("temperature", 0.2),
                "maxOutputTokens": options.get("num_predict", 1200),
            },
        }
        if system_parts:
            payload["systemInstruction"] = {"parts": [{"text": "\n\n".join(system_parts)}]}
        return payload

    @staticmethod
    def _extract_text(data):
        if not isinstance(data, dict):
            return ""
        chunks = []
        for candidate in data.get("candidates", []) or []:
            if not isinstance(candidate, dict):
                continue
            content = candidate.get("content") or {}
            for part in content.get("parts", []) or []:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    chunks.append(part["text"])
            if chunks:
                break
        return "".join(chunks)

    def chat(self, model=None, messages=None, options=None, stream=False, **kwargs):
        if os.getenv("GROWTHLAB_CLOUD_MODE", "").lower() != "true":
            return self.local_client.chat(model=model, messages=messages, options=options,
                                          stream=stream, **kwargs)
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is missing from the environment")
        with self._lock:
            remaining = self.__class__._cooldown_until - time.monotonic()
        if remaining > 0:
            raise GeminiRateLimitError(f"Gemini rate limit cooldown: retry in approximately {remaining:.0f} seconds")

        selected_model = (os.getenv("GEMINI_MODEL") or "gemini-2.5-flash").strip()
        selected_model = selected_model.removeprefix("models/")
        if not selected_model or "/" in selected_model:
            raise ValueError("GEMINI_MODEL must be a Gemini model identifier")
        method = "streamGenerateContent?alt=sse" if stream else "generateContent"
        url = f"{self._base_url}/{quote(selected_model, safe='')}:{method}"
        payload = self._request_payload(messages, options or {})
        response = None
        for attempt in range(1, 4):
            try:
                response = requests.post(
                    url, headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
                    json=payload, stream=bool(stream), timeout=(15, 120),
                )
            except requests.RequestException as exc:
                logging.warning("Gemini network failure: model=%s attempt=%s/3 type=%s",
                                selected_model, attempt, type(exc).__name__)
                if attempt == 3:
                    raise RuntimeError("Gemini network request failed after 3 attempts") from exc
                time.sleep(attempt)
                continue
            status = response.status_code
            if status == 429:
                seconds = self._retry_seconds(response)
                with self._lock:
                    self.__class__._cooldown_until = max(
                        self.__class__._cooldown_until, time.monotonic() + seconds)
                response.close()
                logging.warning("Gemini HTTP 429: model=%s cooldown=%.1fs", selected_model, seconds)
                raise GeminiRateLimitError(f"Gemini rate limit reached; retry after approximately {seconds:.0f} seconds")
            if status in self._transient_statuses:
                detail = self._error_detail(response)
                response.close()
                logging.warning("Gemini temporary HTTP %s: model=%s attempt=%s/3 detail=%s",
                                status, selected_model, attempt, detail)
                if attempt < 3:
                    time.sleep(attempt * 2)
                    continue
                raise RuntimeError(f"Gemini API unavailable (HTTP {status}) after 3 attempts")
            if status >= 400:
                detail = self._error_detail(response)
                response.close()
                logging.error("Gemini API HTTP %s: model=%s detail=%s", status, selected_model, detail)
                raise RuntimeError(f"Gemini API failed (HTTP {status}): {detail}")
            break

        if not stream:
            try:
                data = response.json()
                content = self._extract_text(data)
                if not content:
                    reason = ((data.get("candidates") or [{}])[0].get("finishReason") or "no text")
                    raise RuntimeError(f"Gemini returned no text (finish reason: {reason})")
                return SimpleNamespace(message=SimpleNamespace(content=content))
            finally:
                response.close()

        def generate():
            emitted = False
            try:
                for line in response.iter_lines():
                    if not line:
                        continue
                    if isinstance(line, bytes):
                        line = line.decode("utf-8", errors="replace")
                    if not line.startswith("data:"):
                        continue
                    raw = line[5:].strip()
                    if not raw or raw == "[DONE]":
                        continue
                    try:
                        data = json.loads(raw)
                    except json.JSONDecodeError:
                        logging.warning("Gemini stream contained malformed JSON event")
                        continue
                    if "error" in data:
                        err = data["error"]
                        detail = str(err.get("message", "stream error"))[:200] if isinstance(err, dict) else "stream error"
                        raise RuntimeError(f"Gemini stream failed: {detail}")
                    text = self._extract_text(data)
                    if text:
                        emitted = True
                        yield SimpleNamespace(message=SimpleNamespace(content=text))
                if not emitted:
                    raise RuntimeError("Gemini stream finished without generating text")
            finally:
                response.close()
        return generate()

    def list(self):
        return self.local_client.list()
