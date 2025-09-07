"""Fetch port protocol for the topic evolution system."""

from __future__ import annotations
from typing import Dict, Protocol


class Fetcher(Protocol):
    def fetch(self, url: str) -> Dict: ...
