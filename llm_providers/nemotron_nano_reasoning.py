"""
llm_providers.nemotron_nano_reasoning
─────────────────────────────────────
Tier 1 — NVIDIA Nemotron-3-Nano-Omni 30B Reasoning
Mid-light model with built-in reasoning capabilities.

Provider : NVIDIA Integrate API
Model ID : nvidia/nemotron-3-nano-omni-30b-a3b-reasoning
"""

from __future__ import annotations

import os

from .base import call_nvidia_openai

# ── Configuration ────────────────────────────────────────────────────────────
MODEL_ID = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"
DISPLAY_NAME = "Nemotron Nano 30B Reasoning"
TIER = "Tier 1"
API_KEY_ENV = "NVIDIA_NEMOTRON_NANO_API_KEY"


def get_api_key() -> str:
    return os.getenv(API_KEY_ENV, "").strip()


async def call(prompt: str, api_key: str | None = None) -> str:
    """Send a prompt to Nemotron Nano and return the response text."""
    key = api_key or get_api_key()
    if not key:
        raise Exception(f"No API key configured for {DISPLAY_NAME} ({API_KEY_ENV})")
    
    # We use stream=False internally via call_nvidia_openai, so we won't 
    # stream the reasoning output chunk-by-chunk here, but we pass the
    # required extra_body to enable reasoning correctly on the API side.
    return await call_nvidia_openai(
        model=MODEL_ID,
        prompt=prompt,
        api_key=key,
        temperature=0.6,
        top_p=0.95,
        max_tokens=65536,
        extra_body={
            "chat_template_kwargs": {"enable_thinking": True},
            "reasoning_budget": 16384
        }
    )
