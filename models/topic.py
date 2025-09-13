"""Topic model for the topic evolution system."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
import time

from core_services.math_helpers import Vector
from models.negative_rules import NegativeRules
from models.topic_policy import TopicPolicy


@dataclass
class Topic:
    # Unique ID for this topic (UUID).
    id: str

    # Human-readable name/label for the topic (e.g. "New ML Models").
    name: str

    # Seed terms originally used to search/expand this topic.
    # Also updated when new subtopics are promoted.
    seeds: List[str]

    # Negative rules (block terms/domains/types) to filter irrelevant docs.
    negative: NegativeRules

    # Policy bundle: thresholds for emergence, ranking weights, EMA factors, etc.
    policy: TopicPolicy

    # Long-term centroid vector (weighted incremental mean of all docs ever ingested).
    # This is the "stable identity" of the topic in embedding space.
    centroid_long: Optional[Vector] = None

    # Running total of weights used in centroid_long.
    # Required for correct incremental mean updates.
    doc_count: int = 0

    # Short-term centroid (optional EMA over recent docs).
    # Used to track drift / "what the topic looks like right now".
    centroid_short_ema: Optional[Vector] = None

    # If this topic was spawned from a parent, record parent.topic_id here.
    # Lets you reconstruct the emergence tree (topic graph).
    emerged_from: Optional[str] = None

    # Last time this topic was updated (docs ingested).
    # Used for scheduling refreshes and aging topics out.
    last_updated_ts: float = field(default_factory=time.time)

    # List of child topics.
    children: List[Topic] = field(default_factory=list)
