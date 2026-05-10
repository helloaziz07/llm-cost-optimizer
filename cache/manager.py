"""
Semantic Cache Manager with FAISS Integration
Stores embeddings of past queries and returns cached responses for semantically similar ones.
"""
import faiss
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
from datetime import datetime
import logging

from models import CacheEntry, CacheMetrics
from cache.embedding_service import EmbeddingService
from cache.policy import CacheDecisionPolicy
from config import config

logger = logging.getLogger(__name__)


class SemanticCacheManager:
    """Main cache manager with FAISS-based semantic search and intelligent eviction"""

    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.cache_policy = CacheDecisionPolicy()

        # FAISS index for semantic similarity (Inner Product = cosine for normalized vectors)
        self.index = faiss.IndexFlatIP(config.EMBEDDING_DIM)

        # Cache storage
        self.cache_entries: List[CacheEntry] = []

        # Metrics
        self.metrics = CacheMetrics()

        # Eviction history
        self.eviction_history: List[Dict[str, Any]] = []

        # Dynamic thresholds
        self.current_thresholds = {
            "short": config.THRESHOLD_SHORT_QUERY,
            "medium": config.THRESHOLD_MEDIUM_QUERY,
            "long": config.THRESHOLD_LONG_QUERY,
        }

    def get_adaptive_threshold(self, query: str) -> float:
        """Get adaptive similarity threshold based on query characteristics"""
        query_length = len(query)

        if query_length < config.SHORT_QUERY_MAX_LENGTH:
            return self.current_thresholds["short"]
        elif query_length < config.MEDIUM_QUERY_MAX_LENGTH:
            return self.current_thresholds["medium"]
        else:
            return self.current_thresholds["long"]

    async def search(self, query: str) -> Tuple[Optional[CacheEntry], float, float]:
        """Search for similar cached query"""
        threshold = self.get_adaptive_threshold(query)

        if self.index.ntotal == 0:
            return None, 0.0, threshold

        # Get query embedding
        query_embedding = await self.embedding_service.embed_query(query)
        query_embedding = query_embedding.reshape(1, -1)

        # Search FAISS index
        distances, indices = self.index.search(query_embedding, k=1)

        best_similarity = float(distances[0][0])
        best_index = int(indices[0][0])

        # Check if similarity meets threshold
        if best_similarity >= threshold:
            cache_entry = self.cache_entries[best_index]
            logger.info(f"CACHE HIT: similarity={best_similarity:.4f}, threshold={threshold:.4f}")
            return cache_entry, best_similarity, threshold

        logger.info(f"CACHE MISS: similarity={best_similarity:.4f}, threshold={threshold:.4f}")
        return None, best_similarity, threshold

    async def add(
        self,
        query: str,
        response: str,
        input_tokens: int,
        output_tokens: int,
        cost: float,
        best_similarity: Optional[float] = None
    ) -> bool:
        """Add entry to cache if it passes the decision policy"""
        total_tokens = input_tokens + output_tokens

        # Apply cache decision policy
        should_cache = self.cache_policy.should_cache(
            response_text=response,
            tokens_used=total_tokens,
            best_similarity_score=best_similarity,
            estimated_cost=cost
        )

        if not should_cache:
            return False

        # Check if cache is full and evict if needed
        if len(self.cache_entries) >= config.MAX_CACHE_SIZE:
            self._evict_entries()

        # Get embedding for the query
        query_embedding = await self.embedding_service.embed_query(query)

        # Create cache entry
        entry = CacheEntry(
            query=query,
            response=response,
            embedding=query_embedding.tolist(),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost
        )

        # Add to FAISS index
        self.index.add(query_embedding.reshape(1, -1))

        # Add to cache storage
        self.cache_entries.append(entry)

        # Update metrics
        self.metrics.cache_size = len(self.cache_entries)

        logger.info(f"Added to cache: '{query[:50]}...' (total: {len(self.cache_entries)})")
        return True

    def update_hit(self, entry: CacheEntry, similarity: float, tokens_saved: int, cost_saved: float):
        """Update cache entry on hit"""
        entry.hits += 1
        entry.last_access = datetime.utcnow()

        # Update average similarity (moving average)
        entry.avg_similarity = (
            (entry.avg_similarity * (entry.hits - 1) + similarity) / entry.hits
        )

        entry.llm_tokens_saved += tokens_saved

        # Update metrics
        self.metrics.llm_tokens_saved += tokens_saved
        self.metrics.total_cost_saved += cost_saved

    def _evict_entries(self):
        """Evict low-value entries when cache is full"""
        num_to_evict = max(1, int(len(self.cache_entries) * config.EVICTION_PERCENTAGE))

        logger.info(f"Cache full ({len(self.cache_entries)} entries), evicting {num_to_evict}")

        # Calculate value scores
        now = datetime.utcnow()
        entry_values = []

        for idx, entry in enumerate(self.cache_entries):
            age_seconds = (now - entry.created_at).total_seconds()
            value = self.cache_policy.calculate_cache_value(
                hits=entry.hits,
                age_seconds=age_seconds,
                avg_similarity=entry.avg_similarity,
                tokens_saved=entry.llm_tokens_saved
            )
            entry_values.append((idx, value, entry))

        # Sort by value (ascending) to get lowest value entries
        entry_values.sort(key=lambda x: x[1])

        # Get indices to evict
        indices_to_evict = [idx for idx, _, _ in entry_values[:num_to_evict]]
        indices_to_evict.sort(reverse=True)

        # Log evictions
        for idx in indices_to_evict:
            entry = self.cache_entries[idx]
            age_hours = (now - entry.created_at).total_seconds() / 3600

            self.eviction_history.append({
                "timestamp": now.isoformat(),
                "query": entry.query[:100],
                "response": entry.response[:100],
                "hits": entry.hits,
                "age_hours": round(age_hours, 2),
                "reason": "Low value score"
            })

        # Remove entries
        for idx in indices_to_evict:
            del self.cache_entries[idx]

        # Rebuild FAISS index
        self._rebuild_index()

        # Update metrics
        self.metrics.evictions += num_to_evict
        self.metrics.cache_size = len(self.cache_entries)

    def _rebuild_index(self):
        """Rebuild FAISS index from current cache entries"""
        self.index = faiss.IndexFlatIP(config.EMBEDDING_DIM)

        if len(self.cache_entries) > 0:
            embeddings = np.array([entry.embedding for entry in self.cache_entries], dtype=np.float32)
            self.index.add(embeddings)

    def get_stats(self) -> Dict[str, Any]:
        """Get detailed cache statistics"""
        if len(self.cache_entries) == 0:
            return {
                "total_entries": 0,
                "avg_hits_per_entry": 0,
                "avg_age_seconds": 0,
                "top_queries": [],
            }

        now = datetime.utcnow()
        total_hits = sum(entry.hits for entry in self.cache_entries)
        total_age = sum((now - entry.created_at).total_seconds() for entry in self.cache_entries)

        # Top queries by hits
        sorted_entries = sorted(self.cache_entries, key=lambda e: e.hits, reverse=True)
        top_queries = [
            {
                "query": entry.query[:100],
                "hits": entry.hits,
                "tokens_saved": entry.llm_tokens_saved,
                "avg_similarity": round(entry.avg_similarity, 4),
            }
            for entry in sorted_entries[:5]
        ]

        return {
            "total_entries": len(self.cache_entries),
            "avg_hits_per_entry": round(total_hits / len(self.cache_entries), 2),
            "avg_age_seconds": round(total_age / len(self.cache_entries), 0),
            "top_queries": top_queries,
        }

    def clear(self):
        """Clear all cache entries and reset metrics"""
        self.index = faiss.IndexFlatIP(config.EMBEDDING_DIM)
        self.cache_entries = []
        self.metrics = CacheMetrics()
        self.eviction_history = []
        logger.info("Cache cleared")

    def get_eviction_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent eviction history"""
        return self.eviction_history[-limit:]
