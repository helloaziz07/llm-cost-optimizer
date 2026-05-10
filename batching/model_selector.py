"""
Model Selector
Selects the optimal model based on request characteristics.
Uses the enriched catalog with pricing data for precise cost ranking.
"""
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

from .model_catalog import MODEL_CATALOG


_COST_RANK = {"very-low": 0, "low": 1, "medium": 2, "medium-high": 3, "high": 4}
_LATENCY_RANK = {"low": 0, "medium": 1, "high": 2}


@dataclass(frozen=True)
class SelectionConfig:
    """Model selection configuration"""
    low_min_strength: float = 2.2
    medium_min_strength: float = 2.8
    high_min_strength: float = 4.0
    compliance_bonus: float = 0.6
    diversity_top_n: int = 5


def _normalize_intent(intent_type: Optional[str]) -> str:
    if not intent_type:
        return "general"
    if intent_type in {"coding", "reasoning", "summarization"}:
        return intent_type
    if intent_type == "data_analysis":
        return "reasoning"
    return "general"


def _strength(model: Dict[str, Any], intent: str) -> float:
    s = model.get("strength") or {}
    return float(s.get(intent, s.get("general", 0.0)))


def _actual_cost(model: Dict[str, Any]) -> float:
    """Get actual cost per 1M tokens (input + output) for precise ranking"""
    input_cost = model.get("input_cost_per_1m", 0)
    output_cost = model.get("output_cost_per_1m", 0)
    return input_cost + output_cost


def _rank_latency(model: Dict[str, Any]) -> int:
    return _LATENCY_RANK.get(str(model.get("latency_tier", "medium")), 1)


def _required_strength(complexity_level: Optional[str], cfg: SelectionConfig) -> float:
    if complexity_level == "high":
        return cfg.high_min_strength
    if complexity_level == "medium":
        return cfg.medium_min_strength
    return cfg.low_min_strength


def select_model(
    analysis_json: Dict[str, Any],
    cfg: Optional[SelectionConfig] = None,
    catalog: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[str, Dict[str, Any]]:
    """
    Select optimal model based on request characteristics.
    Returns (model_name, debug_info)
    """
    cfg = cfg or SelectionConfig()
    cat = MODEL_CATALOG if catalog is None else catalog

    intent = _normalize_intent(analysis_json.get("intent_type"))
    complexity = analysis_json.get("complexity_level")
    latency_tol = analysis_json.get("latency_tolerance")
    compliance_needed = bool(analysis_json.get("compliance_needed"))

    required = _required_strength(complexity, cfg)
    if compliance_needed:
        required += cfg.compliance_bonus

    if latency_tol == "low" and complexity in {"medium", "high"}:
        required += 0.2

    # Filter candidates that meet strength threshold
    candidates: List[Dict[str, Any]] = []
    for m in cat:
        if _strength(m, intent) >= required:
            candidates.append(m)

    # Fallback to strongest if none meet threshold
    if not candidates:
        candidates = sorted(cat, key=lambda m: _strength(m, intent), reverse=True)[:5]

    # Rank by actual cost (lowest first), then latency, then strength (highest first)
    ranked: List[Tuple[Tuple[float, float, float], Dict[str, Any]]] = []
    for m in candidates:
        cost = _actual_cost(m)
        lat = float(_rank_latency(m))
        strength = _strength(m, intent)
        ranked.append(((cost, lat, -strength), m))

    ranked.sort(key=lambda x: x[0])

    # Pick from top N for diversity
    top_n = max(1, int(cfg.diversity_top_n))
    top = [m for _, m in ranked[:top_n]]

    # Deterministic choice based on request characteristics
    key = f"{intent}|{complexity}|{latency_tol}|{int(compliance_needed)}"
    idx = sum(ord(c) for c in key) % len(top)
    chosen = top[idx]

    debug = {
        "intent": intent,
        "complexity_level": complexity,
        "latency_tolerance": latency_tol,
        "compliance_needed": compliance_needed,
        "required_strength": round(required, 2),
        "candidates_count": len(candidates),
        "chosen": {
            "name": chosen["name"],
            "provider": chosen.get("provider"),
            "cost_tier": chosen.get("cost_tier"),
            "input_cost_per_1m": chosen.get("input_cost_per_1m"),
            "output_cost_per_1m": chosen.get("output_cost_per_1m"),
            "strength": _strength(chosen, intent),
        },
    }

    return chosen["name"], debug
