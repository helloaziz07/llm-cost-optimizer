"""
Cache Decision Policy
Determines whether to cache responses and calculates entry values for eviction.
"""
from typing import Optional
from config import config
import logging

logger = logging.getLogger(__name__)


class CacheDecisionPolicy:
    """Determines whether to cache a response based on multiple factors"""

    def __init__(self):
        self.min_tokens = config.MIN_TOKENS_TO_CACHE
        self.max_tokens = config.MAX_TOKENS_TO_CACHE
        self.min_cost = config.MIN_COST_TO_CACHE
        self.coverage_threshold = config.SIMILARITY_COVERAGE_THRESHOLD

    def should_cache(
        self,
        response_text: str,
        tokens_used: int,
        best_similarity_score: Optional[float],
        estimated_cost: float
    ) -> bool:
        """Decide whether to cache a response"""

        # Rule 1: Skip very short responses
        if tokens_used < self.min_tokens:
            logger.debug(f"Skip cache: too short ({tokens_used} < {self.min_tokens} tokens)")
            return False

        # Rule 2: Skip very long responses
        if tokens_used > self.max_tokens:
            logger.debug(f"Skip cache: too long ({tokens_used} > {self.max_tokens} tokens)")
            return False

        # Rule 3: Skip cheap responses
        if estimated_cost < self.min_cost:
            logger.debug(f"Skip cache: too cheap (${estimated_cost:.8f} < ${self.min_cost:.8f})")
            return False

        # Rule 4: Skip if very similar coverage already exists
        if best_similarity_score is not None and best_similarity_score >= self.coverage_threshold:
            logger.debug(f"Skip cache: similar coverage exists ({best_similarity_score:.4f})")
            return False

        return True

    def calculate_cache_value(
        self,
        hits: int,
        age_seconds: float,
        avg_similarity: float,
        tokens_saved: int
    ) -> float:
        """Calculate value score for a cache entry (higher is better)"""

        # Frequency component (hits)
        frequency_score = min(hits / 10.0, 1.0)

        # Recency component (inverse of age, with decay)
        max_age = 86400  # 24 hours
        recency_score = max(0, 1.0 - (age_seconds / max_age))

        # Similarity component
        similarity_score = avg_similarity

        # Tokens saved component
        tokens_score = min(tokens_saved / 10000.0, 1.0)

        # Weighted combination
        value = (
            config.WEIGHT_FREQUENCY * frequency_score +
            config.WEIGHT_RECENCY * recency_score +
            config.WEIGHT_SIMILARITY * similarity_score +
            config.WEIGHT_TOKENS_SAVED * tokens_score
        )

        return value
