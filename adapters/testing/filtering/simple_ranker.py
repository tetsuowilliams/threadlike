"""Ranking adapter implementations."""

from __future__ import annotations
from typing import List

from models import Topic, Doc
from protocols import Ranker
from core_services.math_helpers import cos


class SimpleRanker(Ranker):
    """Simple ranking based on similarity and authority."""
    
    def select(self, topic: Topic, docs: List[Doc], K: int) -> List[Doc]:
        if topic.centroid_long is None:
            return sorted(docs, key=lambda d: d.authority, reverse=True)[:K]
        return sorted(docs, key=lambda d: 0.6*cos(d.vec, topic.centroid_long)+0.4*d.authority, reverse=True)[:K]
