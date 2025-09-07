"""Clusterer protocol for the topic evolution system."""

from __future__ import annotations
from typing import List, Protocol

from models import Topic, Doc, ClusterSnapshot


class Clusterer(Protocol):
    def cluster(self, parent: Topic, docs_window: List[Doc]) -> List[ClusterSnapshot]: ...
