"""Deduper protocol for the topic evolution system."""

from __future__ import annotations
from typing import List, Protocol

from models import Doc


class Deduper(Protocol):
    def drop(self, seen_hashes: set[str], docs: List[Doc]) -> List[Doc]: ...
