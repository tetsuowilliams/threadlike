"""Embedding adapter implementations."""

from __future__ import annotations
from typing import List

from protocols import Embedder
from core_services.math_helpers import Vector, norm


class ToyEmbed(Embedder):
    """Simple embedding implementation using hash-based features."""
    
    def __init__(self, dim: int = 32):
        self.dim = dim
    
    def _vec(self, text: str) -> Vector:
        v = [0.0] * self.dim
        for tok in text.lower().split():
            i = (hash(tok) % self.dim)
            v[i] += 1.0
        # l2 normalize
        n = norm(v)
        return [x/n for x in v]
    
    def embed(self, texts: List[str]) -> List[Vector]:
        return [self._vec(t) for t in texts]
