"""
llm_providers.mistral_nemotron
──────────────────────────────
Tier 1 — Mistral Nemotron
Mid-light model for moderate factual and light analytical queries.
Optimized for agentic workflows, coding, and instruction following.

Provider : NVIDIA Integrate API (OpenAI-compatible)
Model ID : mistralai/mistral-nemotron
"""

from __future__ import annotations

import os

from .base import call_nvidia_openai

# ── Configuration ────────────────────────────────────────────────────────────
MODEL_ID = "mistralai/mistral-nemotron"
DISPLAY_NAME = "Mistral Nemotron"
TIER = "Tier 1"
API_KEY_ENV = "NVIDIA_NEMOTRON_API_KEY"


def get_api_key() -> str:
    return os.getenv(API_KEY_ENV, "").strip()


async def call(prompt: str, api_key: str | None = None) -> str:
    """Send a prompt to Mistral Nemotron and return the response text."""
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
