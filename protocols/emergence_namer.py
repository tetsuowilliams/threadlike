"""Emergence namer protocol for the topic evolution system."""

from __future__ import annotations
from typing import List, Tuple, Protocol

from models import Doc


class EmergenceNamer(Protocol):
    def name_and_seeds(self, cluster_docs: List[Doc]) -> Tuple[str, List[str]]: ...
