from __future__ import annotations
import time

from models import Topic, ClusterSnapshot, ClusterState
from protocols import Storage
from core_services.math_helpers import ema_update_vec


class ClusterSmoother:
    """Smooths cluster metrics using exponential moving averages."""
    
    def __init__(self, beta: float):
        self.beta = beta
    
    def update(self, storage: Storage, topic: Topic, snap: ClusterSnapshot, state: ClusterState) -> ClusterState:
        """Update cluster state with smoothed metrics."""
         # Update EMAs
        state.centroid_ema = ema_update_vec(state.centroid_ema, snap.centroid_now, self.beta)
        state.cohesion_ema = (1-self.beta)*state.cohesion_ema + self.beta*snap.cohesion_now
        state.separation_ema = (1-self.beta)*state.separation_ema + self.beta*snap.separation_now
        
        # Check if cluster meets criteria
        meets = (snap.size >= topic.policy.m_min and
                 state.cohesion_ema >= topic.policy.tau_cohesion and
                 (1.0 - state.separation_ema) <= topic.policy.tau_separation)
        
        # Update persistence counter
        state.persistence = (state.persistence + 1) if meets else 0
        state.last_seen_ts = time.time()
        
        # Save updated state
        storage.save_cluster_state(topic.id, state)
        return state