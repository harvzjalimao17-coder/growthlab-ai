"""GrowthLab AI Gemini 3.8 Flash adapter using Google's Interactions API.

Preserves the Ollama-style chat(model, messages, options, stream) contract.
Install: google-genai (add to requirements.txt).
"""
import logging
import os
import time
from types import SimpleNamespace


class GeminiRateLimitError(RuntimeError):
    """Google Gemini quota/rate limit was reached."""


class HybridAIClient:
    def __init__(self, local_client):
        self.local_client = local_client

    @staticmethod
    def _make_input(messages):
        """Build a self-contained turn, preserving supplied conversation history."""
        system = []
        turns = []
        for msg in messages or []:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role") or "user").lower()
            value = msg.get("content") or ""
            if isinstance(value, list):
                value = "\n".join(
                    str(item.get("text", "")) if isinstance(item, dict) else str(item)
                    for item in value
                )
            value = str(value).strip()
            if not value:
                continue
            if role == "system":
                system.append(value)
            elif role in ("user", "assistant"):
                turns.append((role, value))
        if not turns:
            raise ValueError("Gemini requires at least one user or assistant message")
        # A self-contained prompt avoids relying on server-side interaction IDs.
        # It also retains context passed in by main.py, including web/RAG content.
        if len(turns) == 1 and turns[0][0] == "user":
            prompt = turns[0][1]
        else:
            transcript = "\n\n".join(
                f"{'Assistant' if role == 'assistant' else 'User'}: {value}"
                for role, value in turns
            )
            prompt = (
                "Conversation history follows. Reply to the final User message, "
                "using earlier messages only as context.\n\n"
                + transcript + "\n\nAssistant:"
            )
        return "\n\n".join(system), prompt

    @staticmethod
    def _friendly_error(exc):
        message = str(exc)
        lower = message.lower()
        if "429" in lower or "resource_exhausted" in lower or "quota" in lower:
            return GeminiRateLimitError("Gemini rate limit or quota reached; try again later")
        if "503" in lower or "unavailable" in lower or "high demand" in lower:
            return RuntimeError("Gemini 3.8 Flash is temporarily overloaded (HTTP 503); try again later")
        if "404" in lower or "not found" in lower:
            return RuntimeError("Configured Gemini model or API endpoint is unavailable (HTTP 404)")
        if "403" in lower or "permission_denied" in lower:
            return RuntimeError("Gemini API permission denied (HTTP 403); check API key permissions")
        return RuntimeError(f"Gemini Interactions API request failed ({type(exc).__name__})")

    def chat(self, model=None, messages=None, options=None, stream=False, **kwargs):
        if os.getenv("GROWTHLAB_CLOUD_MODE", "").strip().lower() != "true":
            return self.local_client.chat(
                model=model, messages=messages, options=options, stream=stream, **kwargs
            )

        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is missing from Render Environment")

        selected_model = (os.getenv("GEMINI_MODEL") or "gemini-3.8-flash").strip()
        selected_model = selected_model.removeprefix("models/")
        if not selected_model or "/" in selected_model:
            raise ValueError("Invalid GEMINI_MODEL")

        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise RuntimeError(
                "google-genai is not installed; add google-genai to requirements.txt and redeploy"
            ) from exc

        system, prompt = self._make_input(messages)
        options = options or {}
        temperature = options.get("temperature", 0.2)
        max_tokens = options.get("num_predict", 1200)
        try:
            generation_config = {"temperature": float(temperature), "max_output_tokens": int(max_tokens)}
        except (TypeError, ValueError):
            generation_config = {"temperature": 0.2, "max_output_tokens": 1200}

        # Bounded retries before text is emitted; no duplicate streaming output.
        from google.genai import types

        def new_client():
            return genai.Client(api_key=api_key,
                http_options=types.HttpOptions(api_version="v1", timeout=30000))

        params = {"model": selected_model, "input": prompt,
                  "generation_config": generation_config}
        if system:
            params["system_instruction"] = system

        def retryable(exc):
            message = str(exc).lower()
            return any(word in message for word in (
                "503", "429", "high demand", "resource_exhausted",
                "unavailable", "timeout", "timed out", "deadline_exceeded",
                "connection reset", "readtimeout"))

        def log_failure(exc, attempt):
            logging.warning("Gemini request failed: model=%s attempt=%s/3 type=%s",
                            selected_model, attempt, type(exc).__name__)

        if not stream:
            for attempt in range(1, 4):
                client = new_client()
                try:
                    response = client.interactions.create(**params)
                    content = str(getattr(response, "output_text", "") or "")
                    if not content.strip():
                        raise RuntimeError("Gemini returned no text")
                    return SimpleNamespace(message=SimpleNamespace(content=content))
                except Exception as exc:
                    log_failure(exc, attempt)
                    if attempt == 3 or not retryable(exc):
                        raise self._friendly_error(exc) from exc
                    time.sleep(min(2 ** (attempt - 1), 4))
                finally:
                    client.close()

        def generate():
            for attempt in range(1, 4):
                client = new_client()
                emitted = False
                try:
                    events = client.interactions.create(**params, stream=True)
                    for event in events:
                        kind = getattr(event, "event_type", None)
                        if kind == "error":
                            raise RuntimeError("Gemini returned an error event")
                        if kind != "step.delta":
                            continue
                        delta = getattr(event, "delta", None)
                        if getattr(delta, "type", None) != "text":
                            continue
                        content = getattr(delta, "text", "") or ""
                        if content:
                            emitted = True
                            yield SimpleNamespace(message=SimpleNamespace(content=content))
                    if not emitted:
                        raise RuntimeError("Gemini streaming finished without text")
                    return
                except Exception as exc:
                    log_failure(exc, attempt)
                    if emitted or attempt == 3 or not retryable(exc):
                        raise self._friendly_error(exc) from exc
                    time.sleep(min(2 ** (attempt - 1), 4))
                finally:
                    client.close()
        return generate()

    def list(self):
        return self.local_client.list()
