"""
Model Catalog
Comprehensive catalog of LLM models with pricing, capabilities, and performance data.
Enriched with benchmark data from Vellum AI Leaderboard.
"""
from typing import Dict, List, Optional


MODEL_CATALOG: List[Dict] = [
    # =========================================================================
    # Ultra-Cheap / Fast Models
    # =========================================================================
    {
        "name": "gemini-2.0-flash",
        "provider": "google",
        "family": "chat",
        "cost_tier": "very-low",
        "latency_tier": "low",
        "context": 1_000_000,
        "input_cost_per_1m": 0.10,
        "output_cost_per_1m": 0.40,
        "speed_tokens_per_sec": 257,
        "strength": {"coding": 3, "reasoning": 3, "summarization": 3, "general": 3},
    },
    {
        "name": "gpt-4o-mini",
        "provider": "openai",
        "family": "chat",
        "cost_tier": "very-low",
        "latency_tier": "low",
        "context": 128_000,
        "input_cost_per_1m": 0.15,
        "output_cost_per_1m": 0.60,
        "speed_tokens_per_sec": 65,
        "strength": {"coding": 3, "reasoning": 3, "summarization": 3, "general": 3},
    },
    {
        "name": "deepseek-chat",
        "provider": "deepseek",
        "family": "chat",
        "cost_tier": "very-low",
        "latency_tier": "low",
        "context": 32_000,
        "input_cost_per_1m": 0.14,
        "output_cost_per_1m": 0.28,
        "speed_tokens_per_sec": 80,
        "strength": {"coding": 2.5, "reasoning": 3.0, "summarization": 2.5, "general": 2.5},
    },

    # =========================================================================
    # Low-Cost Models
    # =========================================================================
    {
        "name": "gemini-2.5-flash",
        "provider": "google",
        "family": "chat",
        "cost_tier": "low",
        "latency_tier": "low",
        "context": 1_000_000,
        "input_cost_per_1m": 0.15,
        "output_cost_per_1m": 0.60,
        "speed_tokens_per_sec": 200,
        "strength": {"coding": 3.5, "reasoning": 3.5, "summarization": 3.5, "general": 3.5},
    },
    {
        "name": "claude-3.5-haiku",
        "provider": "anthropic",
        "family": "chat",
        "cost_tier": "low",
        "latency_tier": "low",
        "context": 200_000,
        "input_cost_per_1m": 0.80,
        "output_cost_per_1m": 4.00,
        "speed_tokens_per_sec": 66,
        "strength": {"coding": 3, "reasoning": 2.5, "summarization": 3, "general": 3},
    },
    {
        "name": "llama-4-scout",
        "provider": "meta",
        "family": "chat",
        "cost_tier": "low",
        "latency_tier": "low",
        "context": 10_000_000,
        "input_cost_per_1m": 0.11,
        "output_cost_per_1m": 0.34,
        "speed_tokens_per_sec": 2600,
        "strength": {"coding": 3, "reasoning": 3, "summarization": 3, "general": 3},
    },
    {
        "name": "grok-2-mini",
        "provider": "xai",
        "family": "chat",
        "cost_tier": "low",
        "latency_tier": "low",
        "context": 128_000,
        "input_cost_per_1m": 0.30,
        "output_cost_per_1m": 1.00,
        "speed_tokens_per_sec": 100,
        "strength": {"coding": 3.0, "reasoning": 2.8, "summarization": 3.0, "general": 3.0},
    },

    # =========================================================================
    # Medium-Cost Models
    # =========================================================================
    {
        "name": "gpt-4o",
        "provider": "openai",
        "family": "chat",
        "cost_tier": "medium",
        "latency_tier": "medium",
        "context": 128_000,
        "input_cost_per_1m": 2.50,
        "output_cost_per_1m": 10.00,
        "speed_tokens_per_sec": 143,
        "strength": {"coding": 4, "reasoning": 4, "summarization": 4, "general": 4},
    },
    {
        "name": "claude-3.5-sonnet",
        "provider": "anthropic",
        "family": "chat",
        "cost_tier": "medium",
        "latency_tier": "medium",
        "context": 200_000,
        "input_cost_per_1m": 3.00,
        "output_cost_per_1m": 15.00,
        "speed_tokens_per_sec": 78,
        "strength": {"coding": 4, "reasoning": 4, "summarization": 4, "general": 4},
    },
    {
        "name": "deepseek-reasoner",
        "provider": "deepseek",
        "family": "reasoning",
        "cost_tier": "medium",
        "latency_tier": "medium",
        "context": 64_000,
        "input_cost_per_1m": 0.55,
        "output_cost_per_1m": 2.19,
        "speed_tokens_per_sec": 50,
        "strength": {"coding": 3.5, "reasoning": 4.5, "summarization": 3.0, "general": 3.5},
    },
    {
        "name": "grok-2",
        "provider": "xai",
        "family": "chat",
        "cost_tier": "medium",
        "latency_tier": "medium",
        "context": 128_000,
        "input_cost_per_1m": 2.00,
        "output_cost_per_1m": 10.00,
        "speed_tokens_per_sec": 52,
        "strength": {"coding": 3.8, "reasoning": 3.8, "summarization": 3.6, "general": 3.7},
    },

    # =========================================================================
    # Medium-High / Premium Models
    # =========================================================================
    {
        "name": "gemini-2.5-pro",
        "provider": "google",
        "family": "chat",
        "cost_tier": "medium-high",
        "latency_tier": "medium",
        "context": 1_000_000,
        "input_cost_per_1m": 1.25,
        "output_cost_per_1m": 10.00,
        "speed_tokens_per_sec": 191,
        "strength": {"coding": 4.5, "reasoning": 5, "summarization": 4, "general": 4.5},
    },
    {
        "name": "gpt-4.1",
        "provider": "openai",
        "family": "chat",
        "cost_tier": "medium-high",
        "latency_tier": "medium",
        "context": 200_000,
        "input_cost_per_1m": 2.00,
        "output_cost_per_1m": 8.00,
        "speed_tokens_per_sec": 100,
        "strength": {"coding": 5, "reasoning": 5, "summarization": 4, "general": 5},
    },

    # =========================================================================
    # High-Cost / Top-Tier Models
    # =========================================================================
    {
        "name": "claude-sonnet-4",
        "provider": "anthropic",
        "family": "chat",
        "cost_tier": "high",
        "latency_tier": "medium",
        "context": 200_000,
        "input_cost_per_1m": 3.00,
        "output_cost_per_1m": 15.00,
        "speed_tokens_per_sec": 69,
        "strength": {"coding": 5, "reasoning": 5, "summarization": 5, "general": 5},
    },
    {
        "name": "gemini-3-pro",
        "provider": "google",
        "family": "chat",
        "cost_tier": "high",
        "latency_tier": "medium",
        "context": 10_000_000,
        "input_cost_per_1m": 2.00,
        "output_cost_per_1m": 12.00,
        "speed_tokens_per_sec": 128,
        "strength": {"coding": 4.5, "reasoning": 5, "summarization": 4.5, "general": 5},
    },
]


def index_by_name(catalog: Optional[List[Dict]] = None) -> Dict[str, Dict]:
    """Create a name → model dict for quick lookup"""
    cat = MODEL_CATALOG if catalog is None else catalog
    return {m["name"]: m for m in cat}


def get_model_info(model_name: str, catalog: Optional[List[Dict]] = None) -> Optional[Dict]:
    """Get model info by name"""
    return index_by_name(catalog).get(model_name)


def get_model_cost(model_name: str, input_tokens: int, output_tokens: int) -> Optional[float]:
    """Calculate cost for a specific model and token counts"""
    info = get_model_info(model_name)
    if not info:
        return None
    input_cost = (input_tokens / 1_000_000) * info.get("input_cost_per_1m", 0)
    output_cost = (output_tokens / 1_000_000) * info.get("output_cost_per_1m", 0)
    return round(input_cost + output_cost, 8)
