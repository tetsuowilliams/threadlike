"""Search adapter implementations."""

from __future__ import annotations
from typing import List, Dict

from protocols import Search


class ToySearch(Search):
    """Feeds scripted URLs per tick; we ignore the query and use the scenario."""
    
    def __init__(self, scenario):
        self.scenario = scenario
    
    def search(self, query: str, limit: int) -> List[Dict]:
        # scenario provides current tick docs externally; return placeholders; URL is a key
        return [{"url": url} for url in self.scenario.pop_batch()]
