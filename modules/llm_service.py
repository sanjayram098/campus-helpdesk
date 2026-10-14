
"""Groq LLM service for the College Helpdesk RAG system."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from config import GROQ_API_KEY, GROQ_MODEL, GROQ_TIMEOUT


class LLMError(Exception):
    """Raised when the LLM cannot produce an answer."""


@dataclass
class LLMResponse:
    success: bool
    text: str
    error: Optional[str] = None

    @classmethod
    def ok(cls, text: str) -> "LLMResponse":
        return cls(success=True, text=text)

    @classmethod
    def fail(cls, error: str) -> "LLMResponse":
        return cls(success=False, text="", error=error)


class GroqService:
    """Client for Groq chat completions."""

    def __init__(
        self,
        model: str = GROQ_MODEL,
        api_key: str = GROQ_API_KEY,
        timeout: int = GROQ_TIMEOUT,
    ) -> None:
        self.model = (model or "").strip() or GROQ_MODEL
        self.api_key = (api_key or "").strip()
        self.timeout = timeout
        self._client = None

    def has_api_key(self) -> bool:
        return bool(
            self.api_key
            and self.api_key != "your_groq_api_key_here"
        )

    def _get_client(self):
        if not self.has_api_key():
            return None
        if self._client is None:
            from groq import Groq
            self._client = Groq(
                api_key=self.api_key,
                timeout=self.timeout,
                max_retries=0,
            )
        return self._client

    def _format_error(self, exc: Exception) -> str:
        msg = str(exc).lower()
        if "401" in msg or "unauthorized" in msg or "invalid_api_key" in msg:
            return "Invalid Groq API key. Check GROQ_API_KEY in your .env file."
        if "403" in msg or "permission" in msg:
            return "Groq denied access. Check your model and account permissions."
        if "429" in msg or "rate limit" in msg or "quota" in msg:
            return "Groq rate limit reached. Wait for the limit to reset."
        if "404" in msg or "model_not_found" in msg:
            return f"Groq model '{self.model}' is unavailable. Check GROQ_MODEL."
        if "timeout" in msg or "timed out" in msg:
            return "The Groq request timed out. Please try again."
        return "Could not reach Groq. Check your internet, API key, and model."

    def check_connection(self) -> tuple[bool, str]:
        if not self.has_api_key():
            return False, "No Groq API key found. Add GROQ_API_KEY to .env."
        try:
            response = self._get_client().chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Reply with OK."}],
                temperature=0,
                max_tokens=10,
            )
            text = (response.choices[0].message.content or "").strip()
            if not text:
                return False, "Groq returned an empty response."
            return True, f"Connected to Groq using model '{self.model}'."
        except Exception as exc:
            return False, self._format_error(exc)

    def is_reachable(self) -> bool:
        # No network request here; avoids wasting free-tier requests.
        return self.has_api_key()

    def status(self) -> dict:
        has_key = self.has_api_key()
        return {
            "has_key": has_key,
            "reachable": False,
            "model": self.model,
            "message": (
                "Groq key configured. Use Test API Connection to verify it."
                if has_key
                else "No Groq API key found. Add GROQ_API_KEY to .env."
            ),
        }

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
    ) -> LLMResponse:
        if not self.has_api_key():
            return LLMResponse.fail(
                "No Groq API key configured. Add GROQ_API_KEY to .env."
            )
        try:
            response = self._get_client().chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=1200,
            )
            text = (response.choices[0].message.content or "").strip()
            if not text:
                return LLMResponse.fail("The model returned an empty response.")
            return LLMResponse.ok(text)
        except Exception as exc:
            return LLMResponse.fail(self._format_error(exc))


# Keep the old name so the existing RAG pipeline still imports correctly.
GeminiService = GroqService


def get_llm_service() -> GroqService:
    return GroqService()


def check_connection(
    service: Optional[GroqService] = None,
) -> tuple[bool, str]:
    return (service or get_llm_service()).check_connection()