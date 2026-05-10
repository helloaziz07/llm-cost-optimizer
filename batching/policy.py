"""
Batching Policy
Determines batching thresholds and adaptive wait times per model.
"""
from dataclasses import dataclass
from typing import Any, Dict, Optional

from config import config
from .model_catalog import get_model_info


@dataclass(frozen=True)
class BatchingPolicy:
    """Batching thresholds — any of these can close a batch"""
    max_wait_ms: int
    max_batch_size: int
    max_batch_tokens: int


@dataclass(frozen=True)
class AdaptiveBatchingConfig:
    """Adaptive batching configuration"""
    base_wait_ms: int = 80
    min_wait_ms: int = 40
    max_wait_ms: int = 120
    default_max_batch_size: int = 8
    default_max_batch_tokens: int = 3000


def output_length_factor(expected_output_length: Optional[str]) -> float:
    """Approximate output cost/latency multiplier"""
    if expected_output_length == "short":
        return 0.2
    if expected_output_length == "long":
        return 1.2
    return 0.6


def effective_tokens(token_count: int, analysis_json: Dict[str, Any]) -> int:
    """Calculate effective tokens including estimated output"""
    expected = analysis_json.get("expected_output_length")
    factor = output_length_factor(expected)
    eff = int(round(token_count * (1.0 + factor)))
    return max(1, eff)


def adaptive_wait_ms(latency_tolerance: Optional[str], cfg: AdaptiveBatchingConfig) -> int:
    """Adaptive waiting based on latency tolerance"""
    if latency_tolerance == "low":
        wait = 50
    elif latency_tolerance == "high":
        wait = 120
    else:
        wait = cfg.base_wait_ms

    return max(cfg.min_wait_ms, min(cfg.max_wait_ms, wait))


def policy_for_model(
    model_name: str,
    analysis_json: Dict[str, Any],
    cfg: Optional[AdaptiveBatchingConfig] = None,
) -> BatchingPolicy:
    """Return batching thresholds for a request"""
    cfg = cfg or AdaptiveBatchingConfig()

    wait_ms = adaptive_wait_ms(analysis_json.get("latency_tolerance"), cfg)
    max_size = cfg.default_max_batch_size
    max_tokens = cfg.default_max_batch_tokens

    info = get_model_info(model_name)

    if info:
        latency_tier = info.get("latency_tier")
        cost_tier = info.get("cost_tier")

        # Fast models: larger batches
        if latency_tier == "low":
            max_size = max(max_size, 12)
            max_tokens = max(max_tokens, 4500)
            wait_ms = min(wait_ms, 80)

        # Slower models: smaller batches
        if latency_tier == "medium":
            max_size = min(max_size, 8)
            max_tokens = min(max_tokens, 5000)

        # Cheap models: batch more
        if cost_tier in {"very-low", "low"}:
            max_tokens = max(max_tokens, 5000)

        # Expensive models: tighter batches
        if cost_tier == "high":
            max_size = min(max_size, 6)
            max_tokens = min(max_tokens, 3500)

    return BatchingPolicy(max_wait_ms=wait_ms, max_batch_size=max_size, max_batch_tokens=max_tokens)
