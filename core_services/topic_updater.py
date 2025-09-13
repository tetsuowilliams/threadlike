"""Core services for topic evolution (weight-free)."""

from __future__ import annotations
import math
from models import Topic, Doc
from core_services.math_helpers import incremental_mean


class TopicUpdater:
    """
    Updates topic centroids based on new documents.
    Uses a plain incremental mean (no weighting).
    """

    def __init__(self):
        pass  # recency/authority dropped since we're weight-free

    def apply(self, topic: Topic, docs: list[Doc], now_ts: float) -> None:
        """
        Update the topic's long-term centroid with new docs.
        Each doc contributes equally to the mean.
        """
        for d in docs:
            topic.centroid_long, topic.doc_count = incremental_mean(
                topic.centroid_long, getattr(topic, "doc_count", 0), d.vec
            )
        topic.last_updated_ts = now_ts
