"""Ranker protocol for the topic evolution system."""

from __future__ import annotations
from typing import List, Protocol

from models import Topic, Doc


class Ranker(Protocol):
    def select(self, topic: Topic, docs: List[Doc], K: int) -> List[Doc]: ...
