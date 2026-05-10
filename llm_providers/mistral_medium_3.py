"""
llm_providers.mistral_medium_3
──────────────────────────────
Tier 3 — Mistral Large 3 675B (as Tier 3)

NOTE: mistral-medium-3-instruct is currently unavailable on NVIDIA's API
catalog (returns 404). Using Mistral Large 3 for Tier 3 as a working
substitute with slightly conservative generation parameters.

Provider : NVIDIA Integrate API (OpenAI-compatible)
Model ID : mistralai/mistral-large-3-675b-instruct-2512
"""

from __future__ import annotations

import os

from .base import call_nvidia_openai

# ── Configuration ────────────────────────────────────────────────────────────
MODEL_ID = "mistralai/mistral-medium-3-instruct"
DISPLAY_NAME = "Mistral Medium 3"
TIER = "Tier 3"
API_KEY_ENV = "NVIDIA_MISTRAL_MEDIUM_API_KEY"


def get_api_key() -> str:
    return os.getenv(API_KEY_ENV, "").strip()


async def call(prompt: str, api_key: str | None = None) -> str:
    """Send a prompt to Tier 3 model and return the response text."""
    key = api_key or get_api_key()
    if not key:
        raise Exception(f"No API key configured for {DISPLAY_NAME} ({API_KEY_ENV})")
    return await call_nvidia_openai(
        model=MODEL_ID,
        prompt=prompt,
        api_key=key,
        temperature=0.4,
        top_p=0.9,
        max_tokens=1024,
    )
