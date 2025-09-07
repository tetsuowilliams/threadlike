"""Cluster state model for the topic evolution system."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import time

from core_services.math_helpers import Vector


@dataclass
class ClusterState:
    # Cluster identifier, must match the cluster_id from ClusterSnapshot.
    # Used to link "raw snapshot this tick" to "smoothed state across ticks".
    cluster_id: str

    # Exponential Moving Average (EMA) of the cluster centroid.
    # Smooths jitter in centroid_now from tick to tick,
    # so the cluster appears as a stable "drifting point" in embedding space.
    centroid_ema: Optional[Vector] = None

    # EMA of cohesion values over multiple ticks.
    # High sustained cohesion_ema ⇒ cluster is consistently tight,
    # not just a one-off group of similar docs.
    cohesion_ema: float = 0.0

    # EMA of separation (1 - cosine from parent centroid).
    # High sustained separation_ema ⇒ cluster is consistently distinct
    # from the parent topic, not just a temporary outlier.
    separation_ema: float = 0.0

    # Counts how many *consecutive ticks* this cluster has satisfied
    # the promotion thresholds (size, cohesion, separation).
    # Used to enforce persistence before spawning a new topic.
    persistence: int = 0

    # Timestamp of the last tick when this cluster was updated.
    # Lets you expire old cluster states that stop appearing.
    last_seen_ts: float = field(default_factory=time.time)
