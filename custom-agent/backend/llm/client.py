"""
LLM Client — Groq backend.

Architectural contract:
  - This is the ONLY file in the entire codebase that imports groq.
  - It exposes two methods: complete() and complete_json().
  - Zero built-in agents, tool orchestration, or MCP used.
  - Swapping to another provider (OpenAI, Anthropic, etc.) = replacing this file only.
"""
from __future__ import annotations

import json
import os
import re
import time
from typing import Any

from groq import Groq, GroqError
from dotenv import load_dotenv

load_dotenv()

_DEFAULT_MODEL = "openai/gpt-oss-120b"
_MAX_RETRIES = 3
_RETRY_DELAY = 2.0  # seconds; multiplied by attempt index for backoff


class LLMError(Exception):
    """Raised when an LLM request fails after all retries."""


class LLMClient:
    """
    Provider-agnostic LLM interface backed by Groq.

    Public API:
        complete(system, user) -> str
        complete_json(system, user) -> dict

    The agent runtime calls only these two methods.
    All orchestration decisions are made by our Python code, not the SDK.
    """

    def __init__(self, model: str | None = None) -> None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise LLMError(
                "GROQ_API_KEY is not set. "
                "Copy .env.example to .env and add your API key."
            )
        self._client = Groq(api_key=api_key)
        self._model_name = model or os.getenv("GROQ_MODEL", _DEFAULT_MODEL)

        # Usage counters (read by BudgetManager and logger)
        self.total_calls: int = 0
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0

    # ── Public API ──────────────────────────────────────────────────────────

    def complete(self, system: str, user: str) -> str:
        """Send a prompt and return the raw text response."""
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        return self._call_with_retry(messages, response_format={"type": "text"})

    def complete_json(self, system: str, user: str) -> dict[str, Any]:
        """
        Send a prompt expecting a JSON response.
        """
        json_instruction = (
            "\n\nCRITICAL: Respond with ONLY valid JSON. "
            "No explanation. No markdown. No code fences (``` or ```json). "
            "Your entire response must be parseable by json.loads()."
        )
        messages = [
            {"role": "system", "content": system + json_instruction},
            {"role": "user", "content": user},
        ]
        raw = self._call_with_retry(messages, response_format={"type": "json_object"})
        return self._parse_json(raw)

    # ── Internal helpers ────────────────────────────────────────────────────

    def _call_with_retry(self, messages: list[dict], response_format: dict) -> str:
        """Call the API with exponential backoff on transient errors."""
        last_err: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                response = self._client.chat.completions.create(
                    model=self._model_name,
                    messages=messages,
                    response_format=response_format,
                    temperature=0.1,
                    max_tokens=4096,
                    max_completion_tokens=4096,
                )
                self.total_calls += 1

                if response.usage:
                    self.total_input_tokens += response.usage.prompt_tokens or 0
                    self.total_output_tokens += response.usage.completion_tokens or 0

                return response.choices[0].message.content or ""

            except Exception as exc:
                last_err = exc
                if attempt < _MAX_RETRIES - 1:
                    wait = _RETRY_DELAY * (attempt + 1)
                    time.sleep(wait)

        raise LLMError(
            f"LLM call failed after {_MAX_RETRIES} attempts: {last_err}"
        ) from last_err

    @staticmethod
    def _parse_json(raw: str) -> dict[str, Any]:
        """Strip markdown fences and parse JSON. Raises LLMError on failure."""
        cleaned = raw.strip()
        # Remove ```json ... ``` or ``` ... ``` wrappers if present
        cleaned = re.sub(r"^```(?:json)?\s*\n?", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\n?```\s*$", "", cleaned)
        cleaned = cleaned.strip()
        try:
            result = json.loads(cleaned)
            if not isinstance(result, dict):
                raise LLMError(f"Expected JSON object, got {type(result).__name__}.")
            return result
        except json.JSONDecodeError as exc:
            raise LLMError(
                f"LLM returned invalid JSON: {exc}\n"
                f"Raw response (first 500 chars):\n{raw[:500]}"
            ) from exc
