"""Storage adapter implementations."""

from __future__ import annotations
from typing import List, Dict, Tuple, Optional

from models import Topic, Doc, ClusterState
from protocols import Storage
from logging_config import get_logger

logger = get_logger(__name__)


class InMemoryStorage(Storage):
    """In-memory storage implementation for testing."""
    
    def __init__(self):
        self.topics: Dict[str, Topic] = {}
        self.docs_by_topic: Dict[str, List[Doc]] = {}
        self.seen_hashes: Dict[str, set[str]] = {}
        self.cluster_state: Dict[Tuple[str, str], ClusterState] = {}
    
    def load_topic(self, topic_id: str) -> Topic:
        return self.topics[topic_id]
    
    def save_topic(self, topic: Topic) -> None:
        self.topics[topic.id] = topic
        self.docs_by_topic.setdefault(topic.id, [])
        self.seen_hashes.setdefault(topic.id, set())
    
    def save_docs(self, topic_id: str, docs: List[Doc]) -> None:
        self.docs_by_topic.setdefault(topic_id, []).extend(docs)
    
    def recent_docs(self, topic_id: str, window_days: int, limit: int) -> List[Doc]:
        arr = self.docs_by_topic.get(topic_id, [])
        arr = sorted(arr, key=lambda d: d.ts, reverse=True)
        return arr[:limit]
    
    def mark_seen_hashes(self, topic_id: str, hashes: List[str]) -> None:
        self.seen_hashes.setdefault(topic_id, set()).update(hashes)
    
    def seen(self, topic_id: str) -> set[str]:
        return self.seen_hashes.setdefault(topic_id, set())
    
    def load_cluster_state(self, topic_id: str, cluster_id: str) -> Optional[ClusterState]:
        return self.cluster_state.get((topic_id, cluster_id))
    
    def save_cluster_state(self, topic_id: str, state: ClusterState) -> None:
        self.cluster_state[(topic_id, state.cluster_id)] = state
    
    def delete_cluster_state(self, topic_id: str, cluster_id: str) -> None:
        logger.debug(f"Deleting cluster state {cluster_id} for topic {topic_id}")
        self.cluster_state.pop((topic_id, cluster_id), None)

    def get_all_cluster_states_for_topic(self, topic_id: str) -> List[ClusterState]:
        return [st for (tid,_), st in self.cluster_state.items() if tid == topic_id]
