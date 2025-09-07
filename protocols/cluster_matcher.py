"""Cluster matcher protocol for the topic evolution system."""

from __future__ import annotations
from typing import Protocol

from models import Topic, ClusterSnapshot, ClusterState
from protocols.storage import Storage


class ClusterMatcher(Protocol):
    def list_states(self, storage: Storage, topic_id: str) -> list[ClusterState]: ...
    def match_or_create(self, storage: Storage, topic: Topic, snap: ClusterSnapshot) -> ClusterState: ...
    def expire_stale(self, storage: Storage, topic_id: str, max_age_days: int = 90): ...
