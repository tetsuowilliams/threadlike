"""Clusterer protocol for the topic evolution system."""

from __future__ import annotations
from typing import List, Protocol

from models import Topic, Doc, ClusterSnapshot
from core_services.math_helpers import Vector


class Clusterer(Protocol):
    def cluster(self, centroid_long: Vector, docs_window: List[Doc]) -> List[ClusterSnapshot]: ...
