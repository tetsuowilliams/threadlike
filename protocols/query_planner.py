"""Query planner protocol for the topic evolution system."""

from __future__ import annotations
from typing import List, Protocol

from models import Topic


class QueryPlanner(Protocol):
    def plan(self, topic: Topic, k_queries: int) -> List[str]: ...
