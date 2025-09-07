"""Topic policy configuration for emergence detection and ranking."""

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class TopicPolicy:
    # Weight for similarity when ranking candidate docs.
    # Higher = favor docs closer to the topic centroid.
    w_sim: float = 0.6

    # Weight for authority when ranking candidate docs.
    # Higher = favor trusted/authoritative sources.
    w_auth: float = 0.4

    # Lambda parameter for Maximal Marginal Relevance (MMR).
    # Balances relevance vs diversity when selecting top-K docs.
    # 0 → pure relevance, 1 → pure diversity.
    mmr_lambda: float = 0.3

    # Minimum cluster size (# docs) required before considering promotion.
    m_min: int = 6

    # Minimum average cohesion (mean cosine similarity within cluster)
    # required for a cluster to be considered "tight enough".
    tau_cohesion: float = 0.55

    # Maximum cosine similarity allowed between parent topic centroid
    # and cluster centroid. If cos ≤ tau, the cluster is considered
    # "far enough" to be a new subtopic.
    tau_separation: float = 0.70

    # Minimum number of consecutive ticks that a cluster must meet
    # all thresholds before being promoted.
    persistence_min: int = 2

    # Smoothing factor for short-term topic centroid (EMA).
    # Small α = slower drift, large α = more reactive.
    ema_alpha_topic: float = 0.10

    # Smoothing factor for cluster metrics (EMA over cohesion/separation).
    # Prevents one noisy tick from triggering a promotion.
    ema_beta_cluster: float = 0.25
