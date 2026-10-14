"""Central configuration for the College Helpdesk RAG application.

Values are read from environment variables (or a .env file) so that secrets
and machine-specific settings are never hard-coded in the source.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Project root (the folder containing this file).
BASE_DIR: Path = Path(__file__).resolve().parent

# Candidates for .env file to ensure reliable discovery:
# 1. Primary: Project root next to app.py
# 2. Secondary: Desktop workspace directory (if opened from OneDrive Desktop)
_ENV_CANDIDATES = [
    BASE_DIR / ".env",
    Path.home() / "OneDrive" / "Desktop" / "ai based college helpdesk using rag" / ".env",
    Path.home() / "Desktop" / "ai based college helpdesk using rag" / ".env",
]

ENV_FILE: Path = BASE_DIR / ".env"
for _cand in _ENV_CANDIDATES:
    if _cand.exists():
        load_dotenv(_cand, override=True)
        if os.getenv("GEMINI_API_KEY", "").strip():
            ENV_FILE = _cand
            break

# Ambient process environment fallback
load_dotenv(override=False)


def _get_bool(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


# --- Gemini (hosted LLM) ---------------------------------------------------
# Your Google AI Studio API key. Get one at https://aistudio.google.com/apikey
# Store it in a local .env file, NEVER commit it.
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "").strip()

# Default Gemini model used for answer generation.
DEFAULT_GEMINI_MODEL: str = "gemini-3.8-flash"
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL).strip() or DEFAULT_GEMINI_MODEL

# Per-request timeout in seconds.
GEMINI_TIMEOUT: int = int(os.getenv("GEMINI_TIMEOUT", "60"))


def is_gemini_api_key_set() -> bool:
    """Return True if a non-placeholder Gemini API key is configured."""
    key = GEMINI_API_KEY.strip()
    return bool(key and key != "your_gemini_api_key_here")

# --- Groq (hosted LLM) ---
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL: str = os.getenv(
    "GROQ_MODEL", "openai/gpt-oss-20b"
).strip()
GROQ_TIMEOUT: int = int(os.getenv("GROQ_TIMEOUT", "60"))
# --- Embeddings -------------------------------------------------------------
EMBEDDING_MODEL: str = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)

# --- Vector store -----------------------------------------------------------
CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "chroma_db")
CHROMA_COLLECTION_NAME: str = "college_helpdesk"

# --- RAG --------------------------------------------------------------------
RETRIEVAL_TOP_K: int = int(os.getenv("RETRIEVAL_TOP_K", "5"))
RETRIEVAL_THRESHOLD: float = float(os.getenv("RETRIEVAL_THRESHOLD", "0.20"))

# --- PDF processing ---------------------------------------------------------
MAX_PDF_SIZE_MB: int = int(os.getenv("MAX_PDF_SIZE_MB", "50"))
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "120"))

# --- Storage paths ----------------------------------------------------------
DATA_DIR: Path = BASE_DIR / "data"
CHROMA_DIR: Path = BASE_DIR / CHROMA_PERSIST_DIR

# Ensure data and chroma directories exist.
DATA_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

# --- Debug ------------------------------------------------------------------
DEBUG: bool = _get_bool("DEBUG", False)
