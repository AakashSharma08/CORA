"""
llm_providers
─────────────
Central registry for all LLM models used by CORA.

Each model lives in its own file with a consistent interface:
  - MODEL_ID, DISPLAY_NAME, TIER, get_api_key(), call(prompt, api_key?)

This __init__ exports:
  - TIER_MODEL_MAP   — maps tier names to (module, display_name) tuples
  - call_llm()       — routes a prompt to the right model with fallback
  - MODEL_REGISTRY   — ordered list of all registered model modules

Tier Assignments (2026-05 refresh):
  Tier 0  →  Gemma 3n E4B         (lightest, simple queries)
  Tier 1  →  Mistral Nemotron     (moderate factual / light analytical)
  Tier 2  →  Magistral Small 2506 (mid-range analytical / creative)
  Tier 3  →  Mistral Medium 3     (complex reasoning)
  Tier 4  →  Mistral Large 3 675B (hardest multi-step / code)
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

from . import gemma_3n_e4b
from . import mistral_nemotron
from . import magistral_small
from . import mistral_medium_3
from . import mistral_large_3
from .base import close_clients

logger = logging.getLogger("cora.llm")

# ── Model Registry (ordered by tier, lightest → heaviest) ────────────────────
MODEL_REGISTRY = [
    gemma_3n_e4b,       # Tier 0
    mistral_nemotron,   # Tier 1
    magistral_small,    # Tier 2
    mistral_medium_3,   # Tier 3
    mistral_large_3,    # Tier 4
]

# ── Tier → Model Mapping ────────────────────────────────────────────────────
TIER_MODEL_MAP = {
    "Tier 0": gemma_3n_e4b,
    "Tier 1": mistral_nemotron,
    "Tier 2": magistral_small,
    "Tier 3": mistral_medium_3,
    "Tier 4": mistral_large_3,
}

# ── Fallback chain (if the assigned tier's model fails) ──────────────────────
TIER_4_FALLBACKS = [mistral_medium_3, magistral_small, mistral_nemotron, gemma_3n_e4b]


def _build_fallback_chain(primary_module):
    """Build a fallback chain: try every other model in the registry."""
    return [m for m in MODEL_REGISTRY if m is not primary_module]


async def call_llm(
    tier: str,
    prompt: str,
    user_api_key: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Route a prompt to the correct model based on tier assignment.
    Falls back to other models on failure.

    Returns:
        (response_text, display_model_name)
    """
    primary = TIER_MODEL_MAP.get(tier)
    if not primary:
        primary = gemma_3n_e4b  # ultimate fallback

    # Build attempt order: primary first, then Tier 4 fallbacks for T4,
    # or general fallback chain for other tiers
    if tier == "Tier 4":
        attempt_order = [primary] + TIER_4_FALLBACKS
    else:
        attempt_order = [primary] + _build_fallback_chain(primary)

    # Deduplicate while preserving order
    seen = set()
    unique_order = []
    for m in attempt_order:
        if id(m) not in seen:
            seen.add(id(m))
            unique_order.append(m)

    errors = []
    for model_module in unique_order:
        try:
            key = user_api_key or model_module.get_api_key()
            if not key:
                logger.warning(f"No API key for {model_module.DISPLAY_NAME}, skipping.")
                continue

            logger.info(f"Calling {model_module.DISPLAY_NAME} ({model_module.MODEL_ID})")
            text = await model_module.call(prompt, key)
            return text, model_module.DISPLAY_NAME

        except Exception as e:
            logger.error(f"LLM error ({model_module.DISPLAY_NAME}): {e}")
            errors.append(f"{model_module.DISPLAY_NAME}: {str(e)[:200]}")
            continue

    error_detail = " | ".join(errors) if errors else "No API keys configured"
    return f"[Error: {error_detail}]", primary.DISPLAY_NAME


# ── Convenience: get display info for a tier ─────────────────────────────────
def get_tier_model_info(tier: str) -> Tuple[str, str]:
    """Return (model_id, display_name) for the primary model of a tier."""
    module = TIER_MODEL_MAP.get(tier, gemma_3n_e4b)
    return module.MODEL_ID, module.DISPLAY_NAME
