"""Clustering adapter implementations."""

from __future__ import annotations
from typing import List
from models import Topic, Doc, ClusterSnapshot
from protocols import Clusterer
from core_services.math_helpers import Vector, cos, wmean


class KMeans2Clusterer(Clusterer):
    """Tiny k=2 k-means; returns 1–2 clusters depending on spread. Computes cohesion & separation."""
    
    def __init__(self, max_iter=10):
        self.max_iter = max_iter
    
    def cluster(self, parent: Topic, docs_window: list[Doc]) -> list[ClusterSnapshot]:
        if not docs_window: 
            return []
        
        # init: pick two seeds
        v1 = docs_window[0].vec
        v2 = docs_window[-1].vec if len(docs_window) > 1 else v1

        for _ in range(10):
            A, B = [], []
           
            for d in docs_window:
                (A if cos(d.vec, v1) >= cos(d.vec, v2) else B).append(d)
           
            if not A or not B: 
                break
           
            v1, _ = wmean([d.vec for d in A], [d.sample_weight for d in A])
            v2, _ = wmean([d.vec for d in B], [d.sample_weight for d in B])

        snaps = []
        
        for cid, group, vc in [("C0", A, v1), ("C1", B, v2)]:
            if not group: 
                continue
            
            # weighted size is sum of sample_weight (mass), but keep doc_ids for naming
            size_w = sum(d.sample_weight for d in group)
            
            # weighted cohesion = avg cosine(doc, centroid) weighted by sample_weight
            num = sum(d.sample_weight * cos(d.vec, vc) for d in group)
            coh_w = num / max(size_w, 1e-12)
            
            # separation uses parent long-term centroid (unweighted)
            parent_c = parent.centroid_long or vc
            sep = 1.0 - cos(parent_c, vc)
            snaps.append(ClusterSnapshot(
                cluster_id=cid,
                centroid_now=vc,
                size=int(round(size_w)),     # you can keep float if you prefer
                cohesion_now=coh_w,
                separation_now=sep,
                doc_ids=[d.id for d in group],
            ))
        return snaps
    
    def _snap(self, cid: str, centroid: Vector, docs: List[Doc], parent: Topic) -> ClusterSnapshot:
        # cohesion = mean cos to centroid
        sims = [cos(d.vec, centroid) for d in docs]
        cohesion = sum(sims)/len(sims)
        
        # separation = 1 - cos(parent_long, centroid)
        parent_c = parent.centroid_long or centroid
        separation = 1.0 - cos(parent_c, centroid)
        
        return ClusterSnapshot(
            cluster_id=cid, centroid_now=centroid, size=len(docs),
            cohesion_now=cohesion, separation_now=separation,
            doc_ids=[d.id for d in docs]
        )


