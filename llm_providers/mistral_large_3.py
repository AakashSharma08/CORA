"""
llm_providers.mistral_large_3
─────────────────────────────
Tier 3 — Mistral Large 3 (675B)
High-performance model for complex reasoning, code, and multi-step tasks.

Provider : NVIDIA Integrate API (OpenAI-compatible)
Model ID : mistralai/mistral-large-3-675b-instruct-2512
"""

from __future__ import annotations

import os

from .base import call_nvidia_openai

# ── Configuration ────────────────────────────────────────────────────────────
MODEL_ID = "mistralai/mistral-large-3-675b-instruct-2512"
DISPLAY_NAME = "Mistral Large 3"
TIER = "Tier 3"
API_KEY_ENV = "NVIDIA_MISTRAL_LARGE_API_KEY"


def get_api_key() -> str:
    return os.getenv(API_KEY_ENV, "").strip()


async def call(prompt: str, api_key: str | None = None) -> str:
    """Send a prompt to Mistral Large 3 and return the response text."""
    key = api_key or get_api_key()
    if not key:
        raise Exception(f"No API key configured for {DISPLAY_NAME} ({API_KEY_ENV})")
    return await call_nvidia_openai(
        model=MODEL_ID,
        prompt=prompt,
        api_key=key,
        temperature=0.5,
        top_p=1.0,
        max_tokens=2048,
    )
