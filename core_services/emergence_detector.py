from __future__ import annotations
from typing import List
import time
import uuid

from models import Topic, ClusterSnapshot, Doc, PromotionCheck
from protocols import EmergenceNamer
from logging_config import get_logger

logger = get_logger(__name__)


class EmergenceDetector:
    @staticmethod
    def _mass_ok(snap, policy) -> bool:
        return snap.size >= policy.m_min

    @staticmethod
    def _cohesion_ok(state, policy) -> bool:
        return state.cohesion_ema >= policy.tau_cohesion

    @staticmethod
    def _separation_ok(state, policy) -> bool:
        # recall: separation = 1 - cos_parent; we compare the cosine to tau
        cos_parent_ema = 1.0 - state.separation_ema
        return cos_parent_ema <= policy.tau_separation

    @staticmethod
    def _persistence_ok(state, policy) -> bool:
        return state.persistence >= policy.persistence_min

    def explain(self, topic, snap, state) -> PromotionCheck:
        cos_parent_ema = 1.0 - state.separation_ema
        return PromotionCheck(
            mass_ok       = self._mass_ok(snap, topic.policy),
            cohesion_ok   = self._cohesion_ok(state, topic.policy),
            separation_ok = self._separation_ok(state, topic.policy),
            persistence_ok= self._persistence_ok(state, topic.policy),
            size          = snap.size,
            m_min         = topic.policy.m_min,
            cohesion_ema  = state.cohesion_ema,
            tau_cohesion  = topic.policy.tau_cohesion,
            cos_parent_ema= cos_parent_ema,
            tau_separation= topic.policy.tau_separation,
            persistence   = state.persistence,
            persistence_min = topic.policy.persistence_min,
        )

    def ready(self, topic, snap, state) -> bool:
        chk = self.explain(topic, snap, state)
        return chk.ready
    
    def promote(self, parent: Topic, snap: ClusterSnapshot, namer: EmergenceNamer, cluster_docs: List[Doc]) -> Topic:
        """Promote cluster to new topic."""
        name, seeds = namer.name_and_seeds(cluster_docs)
        return Topic(
            id=str(uuid.uuid4()),
            name=name,
            seeds=seeds,
            negative=parent.negative,
            policy=parent.policy,
            centroid_long=snap.centroid_now[:],
            doc_count=snap.size,
            centroid_short_ema=None,
            emerged_from=parent.id,
            last_updated_ts=time.time()
        )
