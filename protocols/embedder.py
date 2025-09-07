"""Embed port protocol for the topic evolution system."""

from __future__ import annotations
from typing import List, Protocol

from core_services.math_helpers import Vector


class Embedder(Protocol):
    def embed(self, texts: List[str]) -> List[Vector]: ...
