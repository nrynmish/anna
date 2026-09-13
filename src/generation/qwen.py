from __future__ import annotations

import json
from typing import Any

import requests

from .prompts import build_messages
from .schemas import GenerationRequest, GenerationResponse


class QwenClient:
    """Thin client for the local llama.cpp OpenAI-compatible server."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8080",
        timeout: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    @property
    def endpoint(self) -> str:
        return f"{self.base_url}/v1/chat/completions"

    def generate(
        self,
        request: GenerationRequest,
        temperature: float = 0.1,
        max_tokens: int = 256,
    ) -> GenerationResponse:
        payload: dict[str, Any] = {
            "messages": build_messages(request),
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }

        response = requests.post(
            self.endpoint,
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()

        body = response.json()
        choices = body.get("choices", [])

        if not choices:
            raise RuntimeError("Qwen server returned no choices.")

        content = choices[0].get("message", {}).get("content", "").strip()

        if not content:
            raise RuntimeError("Qwen server returned an empty response.")

        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Qwen returned non-JSON content: {content!r}"
            ) from exc

        return GenerationResponse.model_validate(parsed)
