"""Cluster snapshot model for the topic evolution system."""

from __future__ import annotations
from dataclasses import dataclass
from typing import List

from core_services.math_helpers import Vector


@dataclass
class ClusterSnapshot:
    # Unique ID for this cluster in the current tick (e.g. "C0", "C1").
    # Used to match against persisted ClusterState across ticks.
    cluster_id: str

    # The *raw* centroid vector of this cluster,
    # computed fresh from the docs in the current sliding window.
    centroid_now: Vector

    # Number of docs assigned to this cluster in the current tick.
    size: int

    # Cohesion = average cosine similarity between docs and centroid_now.
    # High cohesion ⇒ docs are semantically tight (a "real" subtopic).
    cohesion_now: float

    # Separation = 1 - cosine(parent_topic_centroid, centroid_now).
    # High separation ⇒ cluster is far from parent's identity,
    # so it's a candidate to be promoted to a new topic.
    separation_now: float

    # IDs of docs belonging to this cluster.
    # Lets us fetch the actual texts for naming / seed extraction.
    doc_ids: List[str]
