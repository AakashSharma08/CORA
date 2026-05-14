"""
llm_providers.step_3_5_flash
─────────────────────────────
Tier 2 (Fallback) — StepFun Step 3.5 Flash
Sparse MoE model for reasoning, coding, and agentic tasks.

Provider : NVIDIA Integrate API (OpenAI-compatible)
Model ID : stepfun-ai/step-3.5-flash
"""

from __future__ import annotations

import os

from .base import call_nvidia_openai

# ── Configuration ────────────────────────────────────────────────────────────
MODEL_ID = "stepfun-ai/step-3.5-flash"
DISPLAY_NAME = "Step 3.5 Flash"
TIER = "Tier 2"
API_KEY_ENV = "NVIDIA_STEP_FLASH_API_KEY"


def get_api_key() -> str:
    return os.getenv(API_KEY_ENV, "").strip()


async def call(prompt: str, api_key: str | None = None) -> str:
    """Send a prompt to Step 3.5 Flash and return the response text."""
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
