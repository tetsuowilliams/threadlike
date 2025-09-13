"""Core services for topic evolution."""

from __future__ import annotations
import math
from models import Topic, Doc
from core_services.math_helpers import weighted_incremental_mean


class TopicUpdater:
    """Updates topic centroids based on new documents."""
    
    def __init__(self, recency_lambda: float = 0.0):
        self.recency_lambda = recency_lambda

    def _time_weight(self, now_ts: float, doc_ts: float) -> float:
        age_days = max(0.0, (now_ts - doc_ts) / 86400.0)
        return math.exp(-self.recency_lambda * age_days) if self.recency_lambda > 0 else 1.0

    def apply(self, topic: Topic, docs: list[Doc], now_ts: float) -> None:
        for d in docs:
            # total weight = authority * recency_decay 
            w = d.authority * self._time_weight(now_ts, d.ts) 
            topic.centroid_long, topic.weight_sum = weighted_incremental_mean(
                topic.centroid_long, topic.weight_sum, d.vec, w
            )
        topic.last_updated_ts = now_ts