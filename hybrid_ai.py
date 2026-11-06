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

        # Explicit timeout prevents an indefinitely hanging 'Generating answer...' state.
        client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(api_version="v1", timeout=60000),
        )
        params = {
            "model": selected_model,
            "input": prompt,
            "generation_config": generation_config,
        }
        if system:
            params["system_instruction"] = system

        if not stream:
            try:
                interaction = client.interactions.create(**params)
                content = str(getattr(interaction, "output_text", "") or "")
                if not content.strip():
                    raise RuntimeError("Gemini returned no text")
                return SimpleNamespace(message=SimpleNamespace(content=content))
            except Exception as exc:
                logging.error("Gemini Interactions failure: model=%s type=%s", selected_model, type(exc).__name__)
                raise self._friendly_error(exc) from exc
            finally:
                client.close()

        def generate():
            emitted = False
            try:
                events = client.interactions.create(**params, stream=True)
                for event in events:
                    if getattr(event, "event_type", None) != "step.delta":
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
            except Exception as exc:
                logging.error("Gemini Interactions stream failure: model=%s type=%s", selected_model, type(exc).__name__)
                raise self._friendly_error(exc) from exc
            finally:
                client.close()
        return generate()

    def list(self):
        return self.local_client.list()
