"""LLM service for the College Helpdesk RAG system.

This module talks to Google's hosted Gemini API via the official
``google-genai`` SDK. The interface is small and modular so that another
provider (Ollama, OpenAI-compatible, etc.) could be plugged in by adding
another class that implements the same ``generate`` method.

The service never fabricates an answer: if the API call fails, it returns a
clear failure sentinel rather than inventing text.

Privacy note: the retrieved PDF passages (and the user's question) are sent to
Google's servers for inference. The PDFs themselves are processed locally;
only the relevant retrieved chunks are transmitted.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from config import GEMINI_API_KEY, GEMINI_MODEL, GEMINI_TIMEOUT


class LLMError(Exception):
    """Raised when the LLM cannot produce an answer."""


@dataclass
class LLMResponse:
    """Result of an LLM call.

    ``success`` is True only when the model actually returned text.
    """

    success: bool
    text: str
    error: Optional[str] = None

    @classmethod
    def ok(cls, text: str) -> "LLMResponse":
        return cls(success=True, text=text)

    @classmethod
    def fail(cls, error: str) -> "LLMResponse":
        return cls(success=False, text="", error=error)


class GeminiService:
    """Thin client around the Google Gemini API (google-genai SDK)."""

    def __init__(
        self,
        model: str = GEMINI_MODEL,
        api_key: str = GEMINI_API_KEY,
        timeout: int = GEMINI_TIMEOUT,
    ) -> None:
        self.model = (model or "").strip() or GEMINI_MODEL
        self.api_key = (api_key or "").strip()
        self.timeout = timeout
        # The client is created lazily so that imports/tests work even without
        # a valid API key.
        self._client = None

    # -- connectivity / readiness checks -----------------------------------
    def _get_client(self):
        """Create and cache the genai.Client. Returns None if no key."""
        if not self.has_api_key():
            return None
        if self._client is None:
            from google import genai
            from google.genai import types

            # google-genai SDK HttpOptions expects timeout in milliseconds
            timeout_ms = int(self.timeout * 1000) if self.timeout else 60000
            self._client = genai.Client(
                api_key=self.api_key,
                http_options=types.HttpOptions(timeout=timeout_ms),
            )
        return self._client

    def _format_error(self, exc: Exception) -> str:
        """Map common API exceptions to user-friendly messages without leaking secrets."""
        msg = str(exc).lower()
        if "401" in msg or "unauthorized" in msg or "api_key_invalid" in msg or "invalid api key" in msg:
            return (
                "The Gemini API key is invalid or has been revoked. "
                "Check GEMINI_API_KEY in your .env file."
            )
        if "403" in msg or "forbidden" in msg or "permission" in msg:
            return (
                "Permission denied when calling Gemini. Check that the "
                "API key project has access to this model."
            )
        if "429" in msg or "quota" in msg or "rate" in msg or "resource_exhausted" in msg:
            return (
                "Gemini quota or rate limit reached. Please wait a moment "
                "and try again."
            )
        if "404" in msg or "not found" in msg or "no longer available" in msg:
            return (
                f"The model '{self.model}' was not found or is no longer available. "
                "Google recommends 'gemini-3.8-flash'. Check GEMINI_MODEL in your .env file."
            )
        if "503" in msg or "unavailable" in msg or "high demand" in msg:
            return (
                f"The model '{self.model}' is currently experiencing temporary high demand (503). "
                "Please wait a moment and try again."
            )
        if "500" in msg or "internal" in msg:
            return "Google Gemini internal server error (500). Please retry in a moment."
        if "timeout" in msg or "timed out" in msg:
            return "The Gemini request timed out. Please try again."
        return (
            "Could not reach the Gemini API. Check your internet "
            "connection, API key, and model settings."
        )

    def check_connection(self) -> tuple[bool, str]:
        """Perform a live API connection test with a harmless prompt.

        Returns (is_successful, status_or_error_message).
        """
        if not self.has_api_key():
            return (
                False,
                "No Gemini API key found. Add GEMINI_API_KEY to your .env file.",
            )
        client = self._get_client()
        if client is None:
            return False, "Gemini client could not be initialized."
        try:
            response = client.models.generate_content(
                model=self.model,
                contents="ping",
            )
            text = (response.text or "").strip()
            if not text:
                return False, f"Model '{self.model}' responded with empty text."
            return True, f"Successfully connected to Gemini API using model '{self.model}'."
        except Exception as exc:  # noqa: BLE001
            return False, self._format_error(exc)

    def is_reachable(self) -> bool:
        """Return True if the API key is set and the API responds successfully."""
        reachable, _ = self.check_connection()
        return reachable

    def has_api_key(self) -> bool:
        return bool(self.api_key and self.api_key != "your_gemini_api_key_here")

    def status(self) -> dict:
        """Return a status dict for UI display (no secrets)."""
        has_key = self.has_api_key()
        if not has_key:
            return {
                "has_key": False,
                "reachable": False,
                "model": self.model,
                "message": (
                    "No Gemini API key found. Add GEMINI_API_KEY to your .env "
                    "file (see .env.example)."
                ),
            }
        reachable, message = self.check_connection()
        return {
            "has_key": True,
            "reachable": reachable,
            "model": self.model,
            "message": message,
        }

    # -- generation ---------------------------------------------------------
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
    ) -> LLMResponse:
        """Send a system + user prompt to Gemini and return the answer.

        Never raises to the caller; failures are returned as an
        ``LLMResponse`` with ``success=False`` and a safe, user-facing
        error message.
        """
        if not self.has_api_key():
            return LLMResponse.fail(
                "No Gemini API key is configured. Add GEMINI_API_KEY to your "
                ".env file and restart the app."
            )

        client = self._get_client()
        if client is None:
            return LLMResponse.fail(
                "Gemini client could not be initialised."
            )

        try:
            from google.genai import types

            response = client.models.generate_content(
                model=self.model,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=temperature,
                ),
            )
            text = (response.text or "").strip()
            if not text:
                return LLMResponse.fail(
                    "The model returned an empty response. Please try again."
                )
            return LLMResponse.ok(text)
        except Exception as exc:  # noqa: BLE001
            return LLMResponse.fail(self._format_error(exc))


# ---------------------------------------------------------------------------
# Factory / interface
# ---------------------------------------------------------------------------
def get_llm_service() -> GeminiService:
    """Return the default LLM service (Gemini, configured via config.py)."""
    return GeminiService()


def check_connection(service: Optional[GeminiService] = None) -> tuple[bool, str]:
    """Helper function to perform an API connection check on the default or given GeminiService."""
    svc = service or get_llm_service()
    if hasattr(svc, "check_connection"):
        return svc.check_connection()
    if hasattr(svc, "is_reachable"):
        if svc.is_reachable():
            return True, f"Successfully connected to Gemini API using model '{svc.model}'."
        st_info = svc.status()
        return False, st_info.get("message", "Could not connect to Gemini API.")
    return False, "Gemini service is not available."
