"""
Query Tracker
Tracks every query through the pipeline with detailed per-stage metrics.
"""
from typing import Dict, Any, List
from models import QueryTrackingInfo
from config import config
import logging

logger = logging.getLogger(__name__)


class QueryTracker:
    """Manages per-query tracking through the pipeline"""

    def __init__(self):
        self.queries: List[QueryTrackingInfo] = []
        self.max_recent = config.MAX_RECENT_QUERIES

    def add(self, tracking_info: QueryTrackingInfo):
        """Add a completed query to tracking history"""
        self.queries.append(tracking_info)

        # Keep only recent queries
        if len(self.queries) > self.max_recent * 2:
            self.queries = self.queries[-self.max_recent:]

    def get_recent(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent query tracking info"""
        recent = self.queries[-limit:]
        return [q.to_dict() for q in reversed(recent)]

    def get_stats(self) -> Dict[str, Any]:
        """Get overall tracking statistics"""
        if not self.queries:
            return {
                "total_queries": 0,
                "avg_response_time_ms": 0.0,
                "cache_hit_rate": 0.0,
                "total_cost": 0.0,
                "total_cost_saved": 0.0,
            }

        total = len(self.queries)
        avg_time = sum(q.total_time_ms for q in self.queries) / total
        cache_hits = sum(1 for q in self.queries if q.cache_hit)
        total_cost = sum(q.llm_cost for q in self.queries)
        total_saved = sum(q.cost_saved for q in self.queries)

        return {
            "total_queries": total,
            "avg_response_time_ms": round(avg_time, 2),
            "cache_hit_rate": round(cache_hits / total, 4) if total > 0 else 0.0,
            "total_cost": round(total_cost, 6),
            "total_cost_saved": round(total_saved, 6),
        }

    def clear(self):
        """Clear all tracking data"""
        self.queries = []
