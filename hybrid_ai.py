import os
from types import SimpleNamespace

import requests


class HybridAIClient:
    def __init__(self, local_client):
        self.local_client = local_client

    def chat(self, model=None, messages=None, options=None, stream=False, **kwargs):
        if os.getenv("GROWTHLAB_CLOUD_MODE", "").lower() != "true":
            return self.local_client.chat(
                model=model,
                messages=messages,
                options=options,
                stream=stream,
                **kwargs
            )

        api_key = os.getenv("GROQ_API_KEY", "").strip()

        if not api_key:
            raise RuntimeError("GROQ_API_KEY is missing")

        cloud_model = os.getenv(
            "GROQ_MODEL",
            "llama-3.3-70b-versatile"
        )

        payload = {
            "model": cloud_model,
            "messages": messages or [],
            "stream": stream,
            "temperature": (options or {}).get("temperature", 0.2),
            "max_tokens": (options or {}).get("num_predict", 1200)
        }

        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json=payload,
            stream=stream,
            timeout=120
        )

        response.raise_for_status()

        if not stream:
            data = response.json()

            return SimpleNamespace(
                message=SimpleNamespace(
                    content=data["choices"][0]["message"].get("content") or ""
                )
            )

        def generate():
            try:
                for line in response.iter_lines():
                    if not line:
                        continue

                    if not line.startswith(b"data: "):
                        continue

                    raw = line[6:].decode("utf-8")

                    if raw == "[DONE]":
                        break

                    import json
                    data = json.loads(raw)

                    delta = data["choices"][0].get("delta", {})
                    content = delta.get("content") or ""

                    if content:
                        yield SimpleNamespace(
                            message=SimpleNamespace(content=content)
                        )
            finally:
                response.close()

        return generate()

    def list(self):
        return self.local_client.list()
