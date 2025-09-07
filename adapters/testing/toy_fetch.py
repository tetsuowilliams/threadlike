"""Fetch adapter implementations."""

from __future__ import annotations
from typing import Dict

from protocols import Fetcher


class ToyFetch(Fetcher):
    """Fetches documents from a predefined corpus."""
    
    def __init__(self, corpus: Dict[str, Dict]):
        self.corpus = corpus
    
    def fetch(self, url: str) -> Dict:
        return self.corpus[url]
