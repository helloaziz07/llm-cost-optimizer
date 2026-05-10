"""
Cost Optimizer Pipeline — The Central Orchestrator
Connects all modules into one end-to-end flow:
  Prompt Optimization → Semantic Cache → Model Selection → Batching → LLM → Cache Store
"""
import time
import uuid
import logging
from typing import Dict, Any, List, Optional

from models import (
    QueryRequest, QueryResponse, QueryTrackingInfo, PromptAnalysis
)
from config import config
from prompt_optimizer import clean_prompt, shorten_prompt, analyze_complexity, count_tokens
from cache import SemanticCacheManager
from batching import select_model, ModelWiseBatcher
from batching.batcher import BatchRequest
from batching.policy import effective_tokens
from llm import LLMService
from pipeline.tracker import QueryTracker

logger = logging.getLogger(__name__)


class CostOptimizerPipeline:
    """
    Main pipeline that orchestrates all components.

    Flow:
    1. Prompt Optimization (clean → shorten → analyze complexity)
    2. Semantic Cache Lookup (FAISS similarity search)
    3. Model Selection (pick cheapest model meeting requirements)
    4. Batching (group requests by model)
    5. LLM Execution (Gemini API or simulation)
    6. Cache Store (save response for future hits)
    """

    def __init__(self):
        # Initialize all components
        self.cache_manager = SemanticCacheManager()
        self.batcher = ModelWiseBatcher()
        self.llm_service = LLMService()
        self.tracker = QueryTracker()

        logger.info("Pipeline initialized — all components connected")

    async def process_query(self, request: QueryRequest) -> QueryResponse:
        """Process a single query through the entire pipeline"""

        start_time = time.time()
        query_id = str(uuid.uuid4())[:8]

        # Initialize tracking
        tracking = QueryTrackingInfo(
            query_id=query_id,
            original_prompt=request.query,
            optimized_prompt="",
        )

        try:
            # ==================================================================
            # STAGE 1: Prompt Optimization
            # ==================================================================
            opt_start = time.time()

            cleaned = clean_prompt(request.query)
            shortened = shorten_prompt(cleaned)
            analysis = analyze_complexity(shortened)

            original_tokens = count_tokens(request.query)
            optimized_tokens = count_tokens(shortened)

            prompt_analysis = PromptAnalysis(**analysis)

            tracking.optimized_prompt = shortened
            tracking.prompt_analysis = prompt_analysis
            tracking.original_tokens = original_tokens
            tracking.optimized_tokens = optimized_tokens
            tracking.optimization_time_ms = (time.time() - opt_start) * 1000

            logger.info(
                f"[{query_id}] Stage 1 — Prompt Optimized: "
                f"{original_tokens} → {optimized_tokens} tokens "
                f"(intent={analysis['intent_type']}, complexity={analysis['complexity_level']})"
            )

            # ==================================================================
            # STAGE 2: Semantic Cache Lookup
            # ==================================================================
            cache_start = time.time()

            cache_entry, similarity, threshold = await self.cache_manager.search(shortened)

            tracking.cache_similarity_score = similarity
            tracking.cache_threshold_used = threshold
            tracking.cache_lookup_time_ms = (time.time() - cache_start) * 1000

            # CACHE HIT — return cached response directly
            if cache_entry is not None:
                tracking.cache_hit = True
                tracking.llm_response = cache_entry.response
                tracking.cost_saved = cache_entry.cost
                tracking.tokens_saved = cache_entry.input_tokens + cache_entry.output_tokens
                tracking.total_time_ms = (time.time() - start_time) * 1000
                tracking.status = "cache_hit"

                # Update cache hit metrics
                self.cache_manager.update_hit(
                    cache_entry, similarity,
                    tokens_saved=tracking.tokens_saved,
                    cost_saved=tracking.cost_saved
                )
                self.cache_manager.metrics.cache_hits += 1
                self.cache_manager.metrics.total_requests += 1

                self.tracker.add(tracking)

                logger.info(
                    f"[{query_id}] Stage 2 — CACHE HIT (similarity={similarity:.4f}) "
                    f"→ Skipping stages 3-5"
                )

                return QueryResponse(
                    response=cache_entry.response,
                    cached=True,
                    similarity_score=similarity,
                    tokens_used=0,
                    tokens_saved=tracking.tokens_saved,
                    cost=0.0,
                    cost_saved=tracking.cost_saved,
                    latency_ms=tracking.total_time_ms,
                    threshold_used=threshold,
                    selected_model=None,
                    batch_id=None,
                    tracking_id=query_id,
                )

            # CACHE MISS — continue pipeline
            self.cache_manager.metrics.cache_misses += 1
            self.cache_manager.metrics.total_requests += 1

            logger.info(
                f"[{query_id}] Stage 2 — CACHE MISS (similarity={similarity:.4f}, "
                f"threshold={threshold:.4f})"
            )

            # ==================================================================
            # STAGE 3: Model Selection
            # ==================================================================
            selected_model, selection_debug = select_model(analysis)

            tracking.selected_model = selected_model
            tracking.model_selection_reason = str(selection_debug)

            logger.info(
                f"[{query_id}] Stage 3 — Model Selected: {selected_model} "
                f"(intent={selection_debug.get('intent')}, "
                f"candidates={selection_debug.get('candidates_count')})"
            )

            # ==================================================================
            # STAGE 4: Batching (track the request)
            # ==================================================================
            now_ms = int(time.time() * 1000)

            batch_request = BatchRequest(
                request_id=query_id,
                created_at_ms=now_ms,
                optimized_prompt=shortened,
                analysis_json=analysis,
                token_count=optimized_tokens,
                selected_model=selected_model,
                user_id=request.user_id,
            )

            closed_batches = self.batcher.add(batch_request, now_ms=now_ms)

            # Record batch info
            if closed_batches:
                last_batch = closed_batches[-1]
                tracking.batch_id = last_batch.batch_id
                tracking.batch_size = last_batch.size

            logger.info(
                f"[{query_id}] Stage 4 — Batched "
                f"(batch_id={tracking.batch_id}, closed={len(closed_batches)})"
            )

            # ==================================================================
            # STAGE 5: LLM Execution
            # ==================================================================
            llm_start = time.time()

            llm_result = await self.llm_service.generate_response(
                prompt=shortened,
                model_name=selected_model,
                max_tokens=request.max_tokens or 500,
                temperature=request.temperature or 0.7,
            )

            response_text = llm_result["response"]
            input_tokens = llm_result["input_tokens"]
            output_tokens = llm_result["output_tokens"]
            llm_cost = llm_result["cost"]

            tracking.llm_response = response_text
            tracking.llm_input_tokens = input_tokens
            tracking.llm_output_tokens = output_tokens
            tracking.llm_cost = llm_cost
            tracking.llm_response_time_ms = (time.time() - llm_start) * 1000

            # Update cache cost metrics
            self.cache_manager.metrics.llm_tokens_used += input_tokens + output_tokens
            self.cache_manager.metrics.total_cost += llm_cost

            logger.info(
                f"[{query_id}] Stage 5 — LLM Response "
                f"(model={selected_model}, tokens={input_tokens + output_tokens}, "
                f"cost=${llm_cost:.6f}, simulated={llm_result.get('simulated', True)})"
            )

            # ==================================================================
            # STAGE 6: Cache Store (save for future hits)
            # ==================================================================
            cached = await self.cache_manager.add(
                query=shortened,
                response=response_text,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost=llm_cost,
                best_similarity=similarity,
            )

            if cached:
                logger.info(f"[{query_id}] Stage 6 — Response cached for future hits")
            else:
                logger.info(f"[{query_id}] Stage 6 — Response not cached (policy)")

            # Finalize tracking
            tracking.total_time_ms = (time.time() - start_time) * 1000
            tracking.tokens_saved = original_tokens - optimized_tokens
            tracking.status = "completed"

            self.tracker.add(tracking)

            return QueryResponse(
                response=response_text,
                cached=False,
                similarity_score=similarity,
                tokens_used=input_tokens + output_tokens,
                tokens_saved=tracking.tokens_saved,
                cost=llm_cost,
                cost_saved=0.0,
                latency_ms=tracking.total_time_ms,
                threshold_used=threshold,
                selected_model=selected_model,
                batch_id=tracking.batch_id,
                tracking_id=query_id,
            )

        except Exception as e:
            tracking.status = "error"
            tracking.error_message = str(e)
            tracking.total_time_ms = (time.time() - start_time) * 1000
            self.tracker.add(tracking)

            logger.error(f"[{query_id}] Pipeline error: {e}", exc_info=True)
            raise

    def get_system_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system metrics"""
        return {
            "cache": self.cache_manager.metrics.to_dict(),
            "batching": self.batcher.get_stats(),
            "tracking": self.tracker.get_stats(),
            "config": config.to_dict(),
        }

    def get_recent_queries(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent queries"""
        return self.tracker.get_recent(limit)

    def clear_all(self):
        """Clear all data"""
        self.cache_manager.clear()
        self.tracker.clear()
        self.batcher = ModelWiseBatcher()
        logger.info("All pipeline data cleared")
