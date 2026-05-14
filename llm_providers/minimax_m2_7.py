"""
llm_providers.minimax_m2_7
──────────────────────────
Tier 3 (Fallback) — MiniMax M2.7
230B-parameter Sparse MoE model for agentic workflows and complex reasoning.

Provider : NVIDIA Integrate API (OpenAI-compatible)
Model ID : minimaxai/minimax-m2.7
"""

from __future__ import annotations

import os

from .base import call_nvidia_openai

# ── Configuration ────────────────────────────────────────────────────────────
MODEL_ID = "minimaxai/minimax-m2.7"
DISPLAY_NAME = "MiniMax M2.7"
TIER = "Tier 3"
API_KEY_ENV = "NVIDIA_MINIMAX_API_KEY"


def get_api_key() -> str:
    return os.getenv(API_KEY_ENV, "").strip()


async def call(prompt: str, api_key: str | None = None) -> str:
    """Send a prompt to MiniMax M2.7 and return the response text."""
    key = api_key or get_api_key()
    if not key:
        raise Exception(f"No API key configured for {DISPLAY_NAME} ({API_KEY_ENV})")
    return await call_nvidia_openai(
        model=MODEL_ID,
        prompt=prompt,
        api_key=key,
        temperature=0.4,
        top_p=0.9,
        max_tokens=2048,
    )
