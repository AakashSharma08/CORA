"""
llm_providers.magistral_small
─────────────────────────────
Tier 2 — Magistral Small 2506
Mid-range reasoning model for moderately complex analytical and creative tasks.

Provider : NVIDIA Integrate API (OpenAI-compatible SDK)
Model ID : mistralai/magistral-small-2506

NOTE: Magistral is a reasoning model that emits reasoning_content before
the final content. Needs higher max_tokens to accommodate the reasoning
chain before producing the answer.
"""

from __future__ import annotations

import os
import logging
from openai import AsyncOpenAI

logger = logging.getLogger("cora.llm.magistral")

# ── Configuration ────────────────────────────────────────────────────────────
MODEL_ID = "mistralai/magistral-small-2506"
DISPLAY_NAME = "Magistral Small"
TIER = "Tier 2"
API_KEY_ENV = "NVIDIA_MAGISTRAL_API_KEY"
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"

# ── Persistent client (lazy-initialized) ─────────────────────────────────────
_client: AsyncOpenAI | None = None


def get_api_key() -> str:
    return os.getenv(API_KEY_ENV, "").strip()


def _get_client(key: str) -> AsyncOpenAI:
    """Return a persistent AsyncOpenAI client, creating one if needed."""
    global _client
    if _client is None:
        _client = AsyncOpenAI(
            base_url=NVIDIA_BASE_URL,
            api_key=key,
            timeout=120.0,     # Reasoning models need more time
            max_retries=0,
        )
    return _client


async def call(prompt: str, api_key: str | None = None) -> str:
    """Send a prompt to Magistral Small 2506 via OpenAI SDK on NVIDIA endpoint."""
    key = api_key or get_api_key()
    if not key:
        raise Exception(f"No API key configured for {DISPLAY_NAME} ({API_KEY_ENV})")

    client = _get_client(key)

    full_content = ""

    completion = await client.chat.completions.create(
        model=MODEL_ID,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
        top_p=0.95,
        max_tokens=4096,    # Reasoning models need high token budget (reasoning + answer)
        stream=True,
    )

    async for chunk in completion:
        if not getattr(chunk, "choices", None):
            continue
        delta = chunk.choices[0].delta
        # Reasoning models emit reasoning_content (thinking) — we skip it
        # and only capture the final content answer
        if delta.content is not None:
            full_content += delta.content

    return full_content.strip()
