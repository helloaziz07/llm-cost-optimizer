"""
Model-wise Online Batcher
Groups requests by model and manages batch lifecycle.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .policy import AdaptiveBatchingConfig, BatchingPolicy, effective_tokens, policy_for_model


@dataclass
class BatchRequest:
    """A request ready for batching"""
    request_id: str
    created_at_ms: int
    optimized_prompt: str
    analysis_json: Dict[str, Any]
    token_count: int
    selected_model: str
    user_id: Optional[str] = None


@dataclass
class Batch:
    """A batch of requests"""
    batch_id: str
    model_name: str
    created_at_ms: int
    closed_at_ms: Optional[int] = None
    close_reason: Optional[str] = None
    requests: List[BatchRequest] = field(default_factory=list)
    total_effective_tokens: int = 0
    total_input_tokens: int = 0

    @property
    def size(self) -> int:
        return len(self.requests)

    @property
    def max_wait_ms(self) -> int:
        if self.closed_at_ms is None:
            return 0
        return max(0, self.closed_at_ms - self.created_at_ms)


class ModelWiseBatcher:
    """
    Model-wise online batcher.
    - Keeps one open batch per model
    - Uses adaptive waiting based on request characteristics
    - Returns closed batches when thresholds are exceeded
    """

    def __init__(self, cfg: Optional[AdaptiveBatchingConfig] = None):
        self._cfg = cfg or AdaptiveBatchingConfig()
        self._open: Dict[str, Batch] = {}
        self._next_batch_num = 1
        self._closed_batches: List[Batch] = []

    def _new_batch(self, model_name: str, created_at_ms: int) -> Batch:
        batch_id = f"batch-{self._next_batch_num}"
        self._next_batch_num += 1
        return Batch(batch_id=batch_id, model_name=model_name, created_at_ms=created_at_ms)

    def _policy_for_open_batch(self, batch: Batch) -> BatchingPolicy:
        if not batch.requests:
            return BatchingPolicy(
                max_wait_ms=self._cfg.base_wait_ms,
                max_batch_size=self._cfg.default_max_batch_size,
                max_batch_tokens=self._cfg.default_max_batch_tokens
            )
        first = batch.requests[0]
        return policy_for_model(first.selected_model, first.analysis_json, cfg=self._cfg)

    def flush_due(self, now_ms: int) -> List[Batch]:
        """Close any open batches that exceeded max_wait_ms"""
        closed: List[Batch] = []
        for model_name, batch in list(self._open.items()):
            if not batch.requests:
                continue
            pol = self._policy_for_open_batch(batch)
            if now_ms - batch.created_at_ms >= pol.max_wait_ms:
                batch.closed_at_ms = now_ms
                batch.close_reason = "time"
                closed.append(batch)
                self._closed_batches.append(batch)
                del self._open[model_name]
        return closed

    def flush_all(self, now_ms: int) -> List[Batch]:
        """Force-close all open batches"""
        closed: List[Batch] = []
        for model_name, batch in list(self._open.items()):
            if batch.requests:
                batch.closed_at_ms = now_ms
                batch.close_reason = batch.close_reason or "force"
                closed.append(batch)
                self._closed_batches.append(batch)
            del self._open[model_name]
        return closed

    def add(self, req: BatchRequest, now_ms: Optional[int] = None) -> List[Batch]:
        """Add a request and return any closed batches"""
        now_ms = req.created_at_ms if now_ms is None else now_ms

        closed: List[Batch] = []
        closed.extend(self.flush_due(now_ms=now_ms))

        batch = self._open.get(req.selected_model)
        if batch is None:
            batch = self._new_batch(model_name=req.selected_model, created_at_ms=now_ms)
            self._open[req.selected_model] = batch

        pol = policy_for_model(req.selected_model, req.analysis_json, cfg=self._cfg)
        eff_tokens = effective_tokens(req.token_count, req.analysis_json)

        would_exceed_size = batch.size + 1 > pol.max_batch_size
        would_exceed_tokens = batch.total_effective_tokens + eff_tokens > pol.max_batch_tokens

        # If current batch can't accept request, close and create new
        if batch.size > 0 and (would_exceed_size or would_exceed_tokens):
            batch.closed_at_ms = now_ms
            batch.close_reason = "size" if would_exceed_size else "tokens"
            closed.append(batch)
            self._closed_batches.append(batch)

            batch = self._new_batch(model_name=req.selected_model, created_at_ms=now_ms)
            self._open[req.selected_model] = batch

        batch.requests.append(req)
        batch.total_input_tokens += max(0, int(req.token_count))
        batch.total_effective_tokens += eff_tokens

        # Check if batch is now full
        if batch.size >= pol.max_batch_size:
            batch.closed_at_ms = now_ms
            batch.close_reason = "size"
            closed.append(batch)
            self._closed_batches.append(batch)
            del self._open[req.selected_model]
        elif batch.total_effective_tokens >= pol.max_batch_tokens:
            batch.closed_at_ms = now_ms
            batch.close_reason = "tokens"
            closed.append(batch)
            self._closed_batches.append(batch)
            del self._open[req.selected_model]

        return closed

    def get_stats(self) -> Dict[str, Any]:
        """Get batching statistics"""
        total_requests = sum(b.size for b in self._closed_batches)
        batches_by_model: Dict[str, int] = {}

        for batch in self._closed_batches:
            batches_by_model[batch.model_name] = batches_by_model.get(batch.model_name, 0) + 1

        return {
            "total_batches_created": len(self._closed_batches),
            "total_requests_batched": total_requests,
            "avg_batch_size": round(total_requests / len(self._closed_batches), 2) if self._closed_batches else 0,
            "batches_by_model": batches_by_model,
            "open_batches": len(self._open),
        }
