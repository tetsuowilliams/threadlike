"""Search adapter implementations."""

from __future__ import annotations
from typing import List, Dict

from models import Topic
from protocols import Searcher, QueryPlanner


class ToyQueryPlanner(QueryPlanner):
    """Simple query planner that uses topic seeds."""
    
    def plan(self, topic: Topic, k_queries: int) -> List[str]:
        return topic.seeds[:k_queries]