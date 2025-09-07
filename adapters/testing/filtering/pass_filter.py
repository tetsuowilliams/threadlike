"""Filtering adapter implementations."""

from __future__ import annotations
from typing import List

from models import Doc, NegativeRules
from protocols import Filter


class PassFilter(Filter):
    """Filter that passes all documents through."""
    
    def apply(self, negative: NegativeRules, docs: List[Doc]) -> List[Doc]:
        return docs
