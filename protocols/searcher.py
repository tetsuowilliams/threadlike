"""Search port protocol for the topic evolution system."""

from __future__ import annotations
from typing import List, Dict, Protocol


class Searcher(Protocol):
    def search(self, query: str, limit: int) -> List[Dict]: ...
