"""Live LLM client - used both as the mock RAG client's answer generator
and as the optional LLM-as-judge scorer. No offline fallback: every retry
exhausted raises GenerationError instead of returning a degraded answer.
"""

from __future__ import annotations

import time
from typing import Optional

import httpx

from app.utils.config import Config


class GenerationError(RuntimeError):
    """Raised when a live LLM call is required but cannot succeed."""


class LLMJudge:
    """Live OpenRouter chat client, with retry and backoff."""

    def __init__(self, api_key: str, model: str = "openai/gpt-4o-mini"):
        """Initialize the client without making a connection.

        Args:
            api_key: OpenRouter key.
            model: Model slug.
        """
        self.api_key = api_key
        self.model = model
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"

    def judge(self, prompt: str, max_tokens: int = 500) -> str:
        """Call the live LLM, with retry and backoff. No offline fallback.

        Args:
            prompt: The prompt to send.
            max_tokens: Maximum response tokens.

        Returns:
            str: The model's reply.

        Raises:
            GenerationError: No API key is configured, or every retry
                attempt failed.
        """
        if not self.api_key.strip():
            raise GenerationError(
                "OPENROUTER_API_KEY is required - this lab has no offline "
                "fallback for live LLM calls. Set it in .env.")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
        }

        attempts = Config.GENERATION["max_retries"] + 1
        for attempt in range(1, attempts + 1):
            try:
                response = httpx.post(
                    self.base_url, headers=headers, json=data,
                    timeout=Config.GENERATION["llm_timeout_seconds"])
                response.raise_for_status()
                result = response.json()
                content = (result["choices"][0]["message"]["content"] or "").strip()
                if content:
                    return content
            except Exception as exc:  # noqa: BLE001 - retry, then raise
                print(f"[llm_judge] live attempt {attempt}/{attempts} failed: {exc}")
            if attempt < attempts:
                time.sleep(Config.GENERATION["retry_backoff_seconds"] ** attempt)

        raise GenerationError(
            f"live LLM call failed after {attempts} attempt(s) - see the "
            f"printed attempt errors above.")

    def score_relevance(self, question: str, context: str) -> float:
        """Score relevance using the live LLM (0.0-1.0)."""
        prompt = f"""Rate the relevance of this context to the question on a scale of 0.0 to 1.0.

Question: {question}

Context: {context}

Respond with ONLY a number between 0.0 and 1.0."""

        response = self.judge(prompt, max_tokens=10)
        try:
            return float(response.strip())
        except ValueError:
            return 0.5  # malformed reply, not a failed call

    def check_faithfulness(self, answer: str, context: str) -> bool:
        """Check if answer is faithful to context, via the live LLM."""
        prompt = f"""Is this answer faithful to the context? Answer YES or NO only.

Context: {context}

Answer: {answer}

Respond with ONLY 'YES' or 'NO'."""

        response = self.judge(prompt, max_tokens=5)
        return response.strip().upper() == "YES"

    def score_completeness(self, question: str, answer: str) -> float:
        """Score answer completeness (0.0-1.0), via the live LLM."""
        prompt = f"""Rate how completely this answer addresses the question on a scale of 0.0 to 1.0.

Question: {question}

Answer: {answer}

Respond with ONLY a number between 0.0 and 1.0."""

        response = self.judge(prompt, max_tokens=10)
        try:
            return float(response.strip())
        except ValueError:
            return 0.5  # malformed reply, not a failed call
