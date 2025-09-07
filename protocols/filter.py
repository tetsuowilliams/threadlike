"""Filter protocol for the topic evolution system."""

from __future__ import annotations
from typing import List, Protocol

from models import Doc, NegativeRules


class Filter(Protocol):
    def apply(self, negative: NegativeRules, docs: List[Doc]) -> List[Doc]: ...
