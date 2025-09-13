# pip install hdbscan numpy
from typing import List
import numpy as np
import hdbscan
from protocols import Clusterer
from models import Topic, Doc, ClusterSnapshot
from core_services.math_helpers import Vector


class HDBSCANClusterer(Clusterer):
    """
    Density-based clustering with HDBSCAN.
    - Auto-discovers cluster count, labels noise (-1) which we drop.
    - Works well for variable-density corpora and outliers.
    - We run on L2-normalized embeddings so Euclidean ~= cosine.
    """

    def __init__(
        self,
        min_cluster_size: int = 30,   # smallest subtopic mass you care about
        min_samples: int = 15,        # larger -> stricter core definition (more noise)
        cluster_selection_epsilon: float = 0.0,
        n_jobs: int = 8,
        min_mass: int = 10,           # drop clusters smaller than this
        min_cohesion: float = 0.55,   # drop clusters whose cohesion is below this
    ):
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.cluster_selection_epsilon = cluster_selection_epsilon
        self.n_jobs = n_jobs
        self.min_mass = min_mass
        self.min_cohesion = min_cohesion

    def _l2_normalize(self, V: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(V, axis=1, keepdims=True) + 1e-12
        return V / n

    def _centroid(self, X: np.ndarray) -> np.ndarray:
        """Plain mean centroid, renormalized to unit L2."""
        C = X.mean(axis=0)
        C /= (np.linalg.norm(C) + 1e-12)
        return C

    def _cohesion(self, X: np.ndarray, C: np.ndarray) -> float:
        """Average cosine(doc, centroid). With unit vectors, cosine = dot."""
        sims = X @ C  # shape (n,)
        return float(sims.mean())

    def cluster(self, centroid_long: Vector, docs_window: List[Doc]) -> List[ClusterSnapshot]:
        if not docs_window:
            return []

        if len(docs_window) < self.min_cluster_size:
            return []

        # Build matrix of vectors
        V = np.array([d.vec for d in docs_window], dtype=np.float32)

        # Normalize once for cosine behavior with Euclidean metric
        V = self._l2_normalize(V)

        # Fit HDBSCAN
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            cluster_selection_epsilon=self.cluster_selection_epsilon,
            metric="euclidean",
            core_dist_n_jobs=self.n_jobs,
            allow_single_cluster=True,
        )
        labels = clusterer.fit_predict(V)  # -1 = noise

        # Parent centroid (normalized)
        if centroid_long is not None:
            P = np.array(centroid_long, dtype=np.float32)
            P = P / (np.linalg.norm(P) + 1e-12)
        else:
            P = self._centroid(V)

        snaps: List[ClusterSnapshot] = []
        unique_labels = [lab for lab in np.unique(labels) if lab != -1]

        for lab in unique_labels:
            idx = np.where(labels == lab)[0]
            if idx.size == 0:
                continue

            X = V[idx]              # (n_i, d)
            mass = len(idx)
            if mass < self.min_mass:
                continue

            # Plain centroid (unit L2)
            C = self._centroid(X)

            # Plain cohesion (mean cosine to centroid)
            coh = self._cohesion(X, C)
            if coh < self.min_cohesion:
                continue

            # Separation: 1 - cosine(parent, cluster_centroid)
            cos_parent = float(P @ C)
            sep = 1.0 - cos_parent

            snaps.append(ClusterSnapshot(
                cluster_id=f"h{lab}",
                centroid_now=C.tolist(),
                size=mass,
                cohesion_now=coh,
                separation_now=sep,
                doc_ids=[docs_window[i].id for i in idx],
            ))

        return snaps
