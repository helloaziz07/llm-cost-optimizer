"""
Data Models for LLM Cost Optimizer
All Pydantic schemas: prompts, cache, batching, tracking, API contracts.
"""
from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum


class IntentType(str, Enum):
    REASONING = "reasoning"
    SUMMARIZATION = "summarization"
    CODING = "coding"
    DATA_ANALYSIS = "data_analysis"
    CREATIVE_WRITING = "creative_writing"
    FACTUAL_ANSWERING = "factual_answering"
    CONVERSATION = "conversation"
    CLASSIFICATION = "classification"
    GENERAL = "general"
    OTHER = "other"


class ComplexityLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class LatencyTolerance(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# =============================================================================
# Prompt Optimizer Models
# =============================================================================

class PromptAnalysis(BaseModel):
    """Analysis result from prompt optimizer"""
    intent_type: str
    complexity_level: str
    expected_output_length: str
    latency_tolerance: str
    compliance_needed: bool = False


class OptimizedPrompt(BaseModel):
    """Result of prompt optimization"""
    original_prompt: str
    cleaned_prompt: str
    shortened_prompt: str
    analysis: PromptAnalysis
    original_token_count: int
    optimized_token_count: int
    token_reduction_percent: float


# =============================================================================
# Cache Models
# =============================================================================

class CacheEntry(BaseModel):
    """Schema for a cache entry"""
    query: str
    response: str
    embedding: List[float]
    hits: int = 0
    avg_similarity: float = 0.0
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_access: datetime = Field(default_factory=datetime.utcnow)
    llm_tokens_saved: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0

    class Config:
        arbitrary_types_allowed = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "response": self.response,
            "hits": self.hits,
            "avg_similarity": self.avg_similarity,
            "created_at": self.created_at.isoformat(),
            "last_access": self.last_access.isoformat(),
            "llm_tokens_saved": self.llm_tokens_saved,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost": self.cost,
        }


class CacheMetrics(BaseModel):
    """Global cache metrics"""
    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    llm_tokens_used: int = 0
    llm_tokens_saved: int = 0
    total_cost: float = 0.0
    total_cost_saved: float = 0.0
    cache_size: int = 0
    evictions: int = 0

    @property
    def hit_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.cache_hits / self.total_requests

    @property
    def cost_reduction(self) -> float:
        total_potential_cost = self.total_cost + self.total_cost_saved
        if total_potential_cost == 0:
            return 0.0
        return (self.total_cost_saved / total_potential_cost) * 100

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_requests": self.total_requests,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": round(self.hit_rate, 4),
            "llm_tokens_used": self.llm_tokens_used,
            "llm_tokens_saved": self.llm_tokens_saved,
            "total_cost": round(self.total_cost, 6),
            "total_cost_saved": round(self.total_cost_saved, 6),
            "cost_reduction_percent": round(self.cost_reduction, 2),
            "cache_size": self.cache_size,
            "evictions": self.evictions,
        }


# =============================================================================
# Query Tracking Models
# =============================================================================

class QueryTrackingInfo(BaseModel):
    """Complete tracking information for a single query"""
    query_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Original input
    original_prompt: str

    # Prompt optimization stage
    optimized_prompt: str
    prompt_analysis: Optional[PromptAnalysis] = None
    original_tokens: int = 0
    optimized_tokens: int = 0
    optimization_time_ms: float = 0.0

    # Cache stage
    cache_hit: bool = False
    cache_similarity_score: Optional[float] = None
    cache_threshold_used: Optional[float] = None
    cache_lookup_time_ms: float = 0.0

    # Model selection stage
    selected_model: Optional[str] = None
    model_selection_reason: Optional[str] = None

    # Batching stage
    batch_id: Optional[str] = None
    batch_size: Optional[int] = None
    batch_wait_time_ms: Optional[float] = None

    # LLM response stage
    llm_response: str = ""
    llm_input_tokens: int = 0
    llm_output_tokens: int = 0
    llm_cost: float = 0.0
    llm_response_time_ms: float = 0.0

    # Overall metrics
    total_time_ms: float = 0.0
    cost_saved: float = 0.0
    tokens_saved: int = 0

    # Status
    status: str = "completed"
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_id": self.query_id,
            "timestamp": self.timestamp.isoformat(),
            "original_prompt": self.original_prompt[:100] + "..." if len(self.original_prompt) > 100 else self.original_prompt,
            "optimized_prompt": self.optimized_prompt[:100] + "..." if len(self.optimized_prompt) > 100 else self.optimized_prompt,
            "intent_type": self.prompt_analysis.intent_type if self.prompt_analysis else None,
            "complexity": self.prompt_analysis.complexity_level if self.prompt_analysis else None,
            "cache_hit": self.cache_hit,
            "cache_similarity": round(self.cache_similarity_score, 4) if self.cache_similarity_score else None,
            "selected_model": self.selected_model,
            "batch_id": self.batch_id,
            "llm_tokens": self.llm_input_tokens + self.llm_output_tokens,
            "llm_cost": f"${self.llm_cost:.6f}",
            "cost_saved": f"${self.cost_saved:.6f}",
            "total_time_ms": round(self.total_time_ms, 2),
            "status": self.status,
        }


# =============================================================================
# API Request/Response Models
# =============================================================================

class QueryRequest(BaseModel):
    """Request model for query endpoint"""
    query: str
    max_tokens: Optional[int] = 500
    temperature: Optional[float] = 0.7
    user_id: Optional[str] = None


class QueryResponse(BaseModel):
    """Response model for query endpoint"""
    response: str
    cached: bool
    similarity_score: Optional[float] = None
    tokens_used: int
    tokens_saved: int
    cost: float
    cost_saved: float
    latency_ms: float
    threshold_used: float
    selected_model: Optional[str] = None
    batch_id: Optional[str] = None
    tracking_id: str
