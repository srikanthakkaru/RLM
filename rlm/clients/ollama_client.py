import base64
from typing import Any

import requests

from rlm.clients.base_lm import BaseLM
from rlm.core.types import ModelUsageSummary, UsageSummary


class OllamaClient(BaseLM):
    """
    Client for the Ollama backend using the /api/chat endpoint.

    Supports proper system/user/assistant message roles and optional
    image inputs for vision-language models.
    """

    def __init__(
        self,
        model_name: str,
        base_url: str = "http://localhost:11434",
        timeout: int = 1800,
        ollama_options: dict[str, Any] | None = None,
        ollama_format: str | dict[str, Any] | None = None,
        **kwargs,
    ):
        super().__init__(model_name, **kwargs)
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout  # seconds (default 30 min for local models)
        # Ollama runtime options: num_ctx, num_predict, temperature, etc.
        self.ollama_options = ollama_options or {}
        # Top-level structured-output constraint: "json" or a JSON schema dict. When set, Ollama
        # constrains decoding so the response is always valid JSON (or schema-conformant).
        self.ollama_format = ollama_format
        self.total_calls = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.last_input_tokens = 0
        self.last_output_tokens = 0

    def _build_messages(self, prompt: str | list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert prompt to Ollama /api/chat message format."""
        if isinstance(prompt, list) and all(isinstance(m, dict) for m in prompt):
            messages = []
            for msg in prompt:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                entry: dict[str, Any] = {"role": role, "content": content}
                if "images" in msg:
                    entry["images"] = msg["images"]
                messages.append(entry)
            return messages
        if isinstance(prompt, dict):
            return [{"role": "user", "content": str(prompt)}]
        return [{"role": "user", "content": str(prompt)}]

    def completion(self, prompt: str | list[dict[str, Any]], model: str | None = None) -> str:
        """Synchronous chat completion via Ollama /api/chat."""
        messages = self._build_messages(prompt)
        model = model or self.model_name

        try:
            payload: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "stream": False,
            }
            if self.ollama_options:
                payload["options"] = self.ollama_options
            if self.ollama_format is not None:
                payload["format"] = self.ollama_format
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            result = response.json()

            self.total_calls += 1
            self.last_input_tokens = result.get("prompt_eval_count", 0)
            self.last_output_tokens = result.get("eval_count", 0)
            self.total_input_tokens += self.last_input_tokens
            self.total_output_tokens += self.last_output_tokens

            return result.get("message", {}).get("content", "")
        except Exception as e:
            raise RuntimeError(f"Ollama API call failed: {e}") from e

    async def acompletion(self, prompt: str | list[dict[str, Any]], model: str | None = None) -> str:
        """Async completion (delegates to sync for Ollama)."""
        return self.completion(prompt, model=model)

    def get_usage_summary(self) -> UsageSummary:
        return UsageSummary(
            model_usage_summaries={
                self.model_name: ModelUsageSummary(
                    total_calls=self.total_calls,
                    total_input_tokens=self.total_input_tokens,
                    total_output_tokens=self.total_output_tokens,
                )
            }
        )

    def get_last_usage(self) -> ModelUsageSummary:
        return ModelUsageSummary(
            total_calls=1,
            total_input_tokens=self.last_input_tokens,
            total_output_tokens=self.last_output_tokens,
        )


def encode_image_to_base64(image_path: str) -> str:
    """Read an image file and return its base64-encoded string (for Ollama vision)."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")
