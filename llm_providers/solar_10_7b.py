"""
llm_providers.solar_10_7b
─────────────────────────
Tier 0 (Fallback) — Upstage Solar 10.7B Instruct
Lightweight fallback model for simple queries.

Provider : NVIDIA Integrate API (OpenAI-compatible)
Model ID : upstage/solar-10.7b-instruct
"""

from __future__ import annotations

import os

from .base import call_nvidia_openai

# ── Configuration ────────────────────────────────────────────────────────────
MODEL_ID = "upstage/solar-10.7b-instruct"
DISPLAY_NAME = "Solar 10.7B"
TIER = "Tier 0"
API_KEY_ENV = "NVIDIA_SOLAR_API_KEY"


def get_api_key() -> str:
    return os.getenv(API_KEY_ENV, "").strip()


async def call(prompt: str, api_key: str | None = None) -> str:
    """Send a prompt to Solar 10.7B and return the response text."""
    key = api_key or get_api_key()
    if not key:
        raise Exception(f"No API key configured for {DISPLAY_NAME} ({API_KEY_ENV})")
    return await call_nvidia_openai(
        model=MODEL_ID,
        prompt=prompt,
        api_key=key,
        temperature=0.2,
        top_p=0.7,
        max_tokens=1024,
    )
