from __future__ import annotations
import uuid
import time

from models import Topic, ClusterSnapshot, ClusterState
from protocols import Storage, ClusterMatcher
from core_services.math_helpers import cos
from logging_config import get_logger
logger = get_logger(__name__)

class ClusterMatcher(ClusterMatcher):
    def __init__(self, tau_match: float = 0.4, max_age_ticks: int = 6):
        self.tau_match = tau_match
        self.max_age_ticks = max_age_ticks

    def list_states(self, storage: Storage, topic_id: str) -> list[ClusterState]:
        return [st for (tid,_), st in storage.cluster_state.items() if tid == topic_id]

    def match_or_create(self, storage: Storage, topic: Topic, snap: ClusterSnapshot) -> ClusterState:
        logger.debug(f"Matching or creating cluster state for topic {topic.id} with snapshot {snap.cluster_id} Tau match: {self.tau_match}")
        states = self.list_states(storage, topic.id)
        
        # Greedy best match by cosine(sim(snap.centroid_now, state.centroid_ema))
        best, best_sim = None, -1.0

        for st in states:
            if st.centroid_ema is None: 
                continue
            
            s = cos(snap.centroid_now, st.centroid_ema)
            
            if s > best_sim:
                best, best_sim = st, s
        
        logger.debug(f"Best sim for topic {topic.id} with snapshot {snap.cluster_id}: {best_sim}")

        if best is not None and best_sim >= self.tau_match:
            logger.debug(f"Found best match for topic {topic.id} with snapshot {snap.cluster_id}: {best.cluster_id} with similarity {best_sim}")
            return best
        
        # no adequate match → new ephemeral state with a fresh key
        new_key = f"cand_{uuid.uuid4().hex[:8]}"
        logger.debug(f"No adequate cluster state match found for topic {topic.id} with snapshot {snap.cluster_id}, creating new state: {new_key}")
        st = ClusterState(cluster_id=new_key)
        storage.save_cluster_state(topic.id, st)
        return st

    def expire_stale(self, storage: Storage, topic_id: str, max_age_days: int = 90):
        now = time.time()
        max_age_seconds = max_age_days * 86400
        
        for st in self.list_states(storage, topic_id):
            age = now - st.last_seen_ts
            logger.debug(f"Cluster state {st.cluster_id} age: {age} max_age_seconds: {max_age_seconds}")
            if age >= max_age_seconds:
                storage.delete_cluster_state(topic_id, st.cluster_id)
