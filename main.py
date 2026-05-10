"""
FastAPI Application — LLM Cost Optimizer
Entry point: uvicorn main:app --reload --port 8000
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
from typing import Dict, Any

from models import QueryRequest, QueryResponse
from pipeline import CostOptimizerPipeline
from config import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global pipeline instance
pipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown"""
    global pipeline

    logger.info("Starting LLM Cost Optimizer...")
    logger.info(f"Simulation mode — LLM: {config.SIMULATE_LLM}, Embeddings: {config.SIMULATE_EMBEDDINGS}")

    # Initialize pipeline
    pipeline = CostOptimizerPipeline()

    logger.info("System initialized successfully")

    yield

    logger.info("Shutting down...")


app = FastAPI(
    title="LLM Cost Optimizer",
    description="Unified system: Prompt Optimization → Semantic Caching → Model Selection → Batching → LLM",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "system": "LLM Cost Optimizer",
        "version": "1.0.0",
        "simulation_mode": {
            "llm": config.SIMULATE_LLM,
            "embeddings": config.SIMULATE_EMBEDDINGS
        }
    }


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    """
    Main query endpoint — processes query through the entire pipeline

    Flow: Prompt Optimization → Cache Check → Model Selection → Batching → LLM → Cache Store
    """
    logger.info(f"Received query: '{request.query[:100]}...'")

    try:
        response = await pipeline.process_query(request)
        return response
    except Exception as e:
        logger.error(f"Query processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def get_metrics() -> Dict[str, Any]:
    """Get comprehensive system metrics"""
    return pipeline.get_system_metrics()


@app.get("/recent-queries")
async def get_recent_queries(limit: int = 20) -> Dict[str, Any]:
    """Get recent queries with full tracking information"""
    queries = pipeline.get_recent_queries(limit)
    return {
        "count": len(queries),
        "limit": limit,
        "queries": queries
    }


@app.get("/cache/stats")
async def get_cache_stats() -> Dict[str, Any]:
    """Get detailed cache statistics"""
    return {
        "stats": pipeline.cache_manager.get_stats(),
        "metrics": pipeline.cache_manager.metrics.to_dict(),
        "thresholds": pipeline.cache_manager.current_thresholds
    }


@app.get("/cache/entries")
async def get_cache_entries(limit: int = 20) -> Dict[str, Any]:
    """Get cache entries"""
    entries = [
        {
            "query": entry.query[:100],
            "response": entry.response[:100],
            "hits": entry.hits,
            "avg_similarity": round(entry.avg_similarity, 4),
            "tokens_saved": entry.llm_tokens_saved,
            "created_at": entry.created_at.isoformat(),
        }
        for entry in pipeline.cache_manager.cache_entries[:limit]
    ]

    return {
        "total_entries": len(pipeline.cache_manager.cache_entries),
        "showing": len(entries),
        "entries": entries
    }


@app.get("/cache/evictions")
async def get_eviction_history(limit: int = 50) -> Dict[str, Any]:
    """Get eviction history"""
    history = pipeline.cache_manager.get_eviction_history(limit)
    return {
        "total_evictions": pipeline.cache_manager.metrics.evictions,
        "history_count": len(history),
        "evictions": history
    }


@app.post("/cache/clear")
async def clear_cache() -> Dict[str, str]:
    """Clear all cache entries"""
    pipeline.cache_manager.clear()
    return {"status": "success", "message": "Cache cleared"}


@app.get("/batching/stats")
async def get_batching_stats() -> Dict[str, Any]:
    """Get batching statistics"""
    return pipeline.batcher.get_stats()


@app.get("/config")
async def get_config() -> Dict[str, Any]:
    """Get current configuration"""
    return config.to_dict()


@app.post("/clear-all")
async def clear_all() -> Dict[str, str]:
    """Clear all data (cache, tracking, batching)"""
    pipeline.clear_all()
    return {"status": "success", "message": "All data cleared"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
