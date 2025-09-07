"""Document model for the topic evolution system."""

from __future__ import annotations
from dataclasses import dataclass

from core_services.math_helpers import Vector


@dataclass
class Doc:
    # Unique ID for this document (internal UUID).
    id: str

    # Timestamp (epoch seconds) when this doc was published or ingested.
    # Used for recency weighting and sliding window clustering.
    ts: float

    # Canonical URL of the document (if available).
    url: str

    # Domain/host of the source (e.g. "arxiv.org", "medium.com").
    # Useful for filtering with negative rules or authority heuristics.
    domain: str

    # Title of the doc (page title, paper title, etc).
    title: str

    # Full text (or extracted text) of the doc.
    # Used for embedding, seed-term generation, or summarization.
    text: str

    # Type/category of source (e.g. "paper", "repo", "blog", "news").
    # Lets you assign different authority weights or filtering rules.
    dtype: str

    # Authority score in [0..1] — higher = more trustworthy/important.
    # Could come from source reputation, citation count, pagerank, etc.
    authority: float

    # Embedding vector of the document text.
    # Basis for clustering, centroid updates, and similarity search.
    vec: Vector

    # Content hash (e.g. SHA256 of text) used for deduplication.
    hash: str

    # Arm/experiment identifier (if A/B testing or multiple retrieval arms).
    # Lets you trace which retrieval strategy brought in this doc.
    arm_id: str
    
    # NEW: multiplicity after within-tick dedup aggregation
    sample_weight: float = 1.0  # counts how many identical records were collapsed
