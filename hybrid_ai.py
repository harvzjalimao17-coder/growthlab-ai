"""GrowthLab AI adapter for local Ollama or Google's Gemini Interactions API.

Preserves the Ollama-compatible chat() response shape expected by main.py.
Uses the documented v1 Interactions endpoint, without an extra SDK dependency.
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
    """Gemini rate limit or an active provider cooldown."""


class HybridAIClient:
    _lock = threading.Lock()
    _cooldown_until = 0.0
    _endpoint = "https://generativelanguage.googleapis.com/v1/interactions"
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
    def _input_and_system(messages):
        """Keep main.py's message history without relying on server-side storage.

        System messages remain system instructions; conversational turns are
        clearly labeled inside a single input string for stateless operation.
        """
        system_parts = []
        turns = []
        for message in messages or []:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role") or "user").lower()
            content = message.get("content") or ""
            if isinstance(content, list):
                content = "\n".join(str(part.get("text", "")) if isinstance(part, dict) else str(part) for part in content)
            content = str(content).strip()
            if not content:
                continue
            if role == "system":
                system_parts.append(content)
            elif role in ("user", "assistant"):
                turns.append((role, content))
        if not turns:
            raise ValueError("Gemini requires at least one user or assistant message")
        if len(turns) == 1 and turns[0][0] == "user":
            prompt = turns[0][1]
        else:
            prompt = "Conversation history (chronological):\n" + "\n\n".join(
                f"{role.upper()}:\n{content}" for role, content in turns
            ) + "\n\nRespond to the latest USER message, using earlier turns only as context."
        return prompt, "\n\n".join(system_parts)

    @staticmethod
    def _error_detail(response):
        try:
            error = response.json().get("error", {})
            if isinstance(error, dict):
                return str(error.get("message") or error.get("status") or "No structured provider error returned")[:240]
        except (ValueError, AttributeError, TypeError):
            pass
        return "No structured provider error returned"

    @staticmethod
    def _output_text(data):
        # Interactions responses use steps[].content[].text, not choices[].
        parts = []
        for step in data.get("steps", []) or []:
            if not isinstance(step, dict) or step.get("type") != "model_output":
                continue
            for content in step.get("content", []) or []:
                if isinstance(content, dict) and content.get("type") == "text":
                    parts.append(str(content.get("text") or ""))
        return "".join(parts)

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
        selected_model = os.getenv("GEMINI_MODEL", "gemini-3.8-flash").strip() or "gemini-3.8-flash"
        prompt, system_instruction = self._input_and_system(messages)
        payload = {
            "model": selected_model,
            "input": prompt,
            "stream": bool(stream),
            "store": False,
            "generation_config": {
                "temperature": options.get("temperature", 0.2),
                "max_output_tokens": options.get("num_predict", 1200),
            },
        }
        if system_instruction:
            payload["system_instruction"] = system_instruction

        response = None
        for attempt in range(1, 4):
            try:
                response = requests.post(
                    self._endpoint,
                    headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
                    json=payload, stream=bool(stream), timeout=(15, 120),
                )
            except requests.RequestException as exc:
                if attempt == 3:
                    raise RuntimeError("Gemini network request failed after 3 attempts") from exc
                logging.warning("Gemini connection error: model=%s attempt=%s/3", selected_model, attempt)
                time.sleep(attempt)
                continue
            status = response.status_code
            if status == 429:
                seconds = self._retry_seconds(response)
                with self._lock:
                    self.__class__._cooldown_until = max(
                        self.__class__._cooldown_until, time.monotonic() + seconds
                    )
                response.close()
                logging.warning("Gemini HTTP 429: model=%s cooldown=%.1fs", selected_model, seconds)
                raise GeminiRateLimitError(f"Gemini rate limit reached; retry after approximately {seconds:.0f} seconds.")
            if status in self._transient_statuses:
                detail = self._error_detail(response)
                response.close()
                logging.warning("Gemini temporary HTTP %s: model=%s attempt=%s/3 detail=%s",
                                status, selected_model, attempt, detail)
                if attempt < 3:
                    time.sleep(attempt * 2)
                    continue
                raise RuntimeError(f"Gemini Interactions API failed (HTTP {status}) after 3 attempts")
            if status >= 400:
                detail = self._error_detail(response)
                response.close()
                logging.error("Gemini Interactions API HTTP %s: model=%s detail=%s", status, selected_model, detail)
                raise RuntimeError(f"Gemini Interactions API failed (HTTP {status}): {detail}")
            break

        if not stream:
            try:
                data = response.json()
                if data.get("status") == "failed":
                    raise RuntimeError("Gemini interaction reported failure")
                content = self._output_text(data)
                if not content:
                    raise RuntimeError("Gemini interaction returned no text")
                return SimpleNamespace(message=SimpleNamespace(content=content))
            finally:
                response.close()

        def generate():
            try:
                # Interactions streams SSE with event_type='step.delta',
                # delta.type='text', delta.text='<chunk>'.
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
                        event = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    if event.get("event_type") == "interaction.status_update" and event.get("status") == "failed":
                        raise RuntimeError("Gemini streaming interaction failed")
                    if event.get("error"):
                        raise RuntimeError("Gemini streaming interaction returned an error")
                    if event.get("event_type") == "step.delta":
                        delta = event.get("delta") or {}
                        if delta.get("type") == "text" and delta.get("text"):
                            yield SimpleNamespace(message=SimpleNamespace(content=delta["text"]))
            finally:
                response.close()

        return generate()

    def list(self):
        return self.local_client.list()
