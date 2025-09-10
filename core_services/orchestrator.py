"""Orchestrator for the topic evolution system.

This module wires together the "ports" (interfaces/adapters) and the core
domain services to perform one refresh cycle ("tick") for a Topic.

High-level flow per tick:
  1) Plan queries from the current Topic state (seeds/centroids/etc).
  2) Harvest candidate pages (search → fetch → embed).
  3) Filter, deduplicate, rank (optionally with MMR) and keep top-K docs.
  4) Update the Topic's durable identity:
        - long-term centroid via weighted incremental mean
        - (optionally) short-term EMA centroid for drift/UX
     Persist docs and Topic state (idempotent, restart-safe).
  5) Re-cluster the recent sliding window to observe *current* structure.
  6) For each raw cluster snapshot:
        - match to a persisted ephemeral ClusterState (by centroid similarity),
        - smooth metrics with EMA (centroid/cohesion/separation),
        - update a persistence counter (consecutive ticks above thresholds),
        - if ready → promote to a child Topic (and clear the ephemeral state).
  7) Expire stale ephemeral clusters (housekeeping).
"""

from __future__ import annotations

"""
Design notes:
- The orchestrator stays *framework-agnostic*: all infra is behind ports.
- No global state: everything needed is passed in or loaded via storage.
- Deterministic & idempotent where possible (e.g., dedup by content hash).
"""

from typing import List, Dict
import time
import uuid

# Domain data models (pure dataclasses, no infra types)
from models import Topic, Doc

# Ports (interfaces) the orchestrator depends on. Concrete adapters live elsewhere.
from protocols import (
    QueryPlanner,       # builds concrete search queries from a Topic
    Searcher,         # executes search against a web/API index
    Fetcher,          # fetches page content/metadata for a URL
    Embedder,          # converts text → vector embedding
    Storage,        # durable storage for topics/docs/ephemeral clusters
    Ranker,             # scores + diversifies candidate docs
    Filter,             # applies NegativeRules (terms/domains/types)
    Deduper,            # removes exact/near duplicates via hash/sketches
    Clusterer,          # clusters recent-window docs into raw snapshots
    EmergenceNamer,     # converts a cluster’s docs → human label + seeds
    ClusterMatcher, # matches raw snapshots ↔ persisted ClusterState
)

# Core domain services (pure logic; stateless aside from Topic/ClusterState mutation)
from core_services.topic_updater import TopicUpdater           # long-term mean (+ optional short-term EMA)
from core_services.cluster_smoother import ClusterSmoother     # EMA for cluster metrics + persistence counter
from core_services.emergence_detector import EmergenceDetector # promotion rule evaluation
from logging_config import get_logger

logger = get_logger("orchestrator")

class Orchestrator:
    """Main orchestrator that coordinates all components for topic evolution.

    This class is intentionally thin: it sequences calls, handles glue logic,
    and persists/returns results. All policy/math lives in the injected services.

    Parameters (ports/services):
      - planner: builds concrete queries from Topic (seeds + expansion).
      - search: executes those queries (Serper/Bing/etc).
      - fetcher: loads text + metadata for each hit (title, ts, domain, etc).
      - embedder: maps text to a fixed-dimension vector (encoder-agnostic).
      - storage: abstracts persistence (topics/docs/cluster-state).
      - ranker: scores docs (relevance/authority/recency) + applies diversity (MMR).
      - filtr: NegativeRules filter (block terms/domains/types).
      - deduper: content-hash/near-dup elimination.
      - clusterer: clusters *recent window* docs → raw ClusterSnapshot(s).
      - smoother: maintains EMA metrics & persistence for clusters across ticks.
      - updater: updates Topic’s long-term centroid with new docs this tick.
      - emergence: checks smoothed metrics + persistence for promotion.
      - namer: extracts readable label + seeds for a promoted child topic.
      - matcher: finds/creates ephemeral ClusterState for each raw snapshot.

    Policy knobs:
      - window_days: how far back to look when clustering (recency window).
      - K_queries: how many queries to issue per tick.
      - K_keep: how many newly-harvested docs to accept per tick after ranker.
    """

    def __init__(
        self,
        planner: QueryPlanner,
        searcher: Searcher,
        fetcher: Fetcher,
        embedder: Embedder,
        storage: Storage,
        ranker: Ranker,
        filtr: Filter,
        deduper: Deduper,
        clusterer: Clusterer,
        smoother: ClusterSmoother,
        updater: TopicUpdater,
        emergence: EmergenceDetector,
        namer: EmergenceNamer,
        matcher: ClusterMatcher,
        window_days: int = 30,
        K_queries: int = 6,
        K_keep: int = 20,
    ):
        # Adapters / services
        self.planner = planner
        self.searcher = searcher
        self.fetcher = fetcher
        self.embedder = embedder
        self.storage = storage
        self.ranker = ranker
        self.filter = filtr
        self.deduper = deduper
        self.clusterer = clusterer
        self.smoother = smoother
        self.updater = updater
        self.emergence = emergence
        self.namer = namer
        self.matcher = matcher
        
        # Policy
        self.window_days = window_days
        self.K_queries = K_queries
        self.K_keep = K_keep

    def tick(self, topic_id: str) -> Dict:
        """
        Execute one end-to-end refresh for a single Topic.

        Steps:
          1) Plan queries → search → fetch → embed.
          2) Filter / dedup / rank → accept top-K.
          3) Update Topic's long-term centroid with *new docs* (not whole window).
          4) Cluster recent window → raw cluster snapshots.
          5) For each snapshot: match-or-create ephemeral state → EMA update → persistence update.
          6) If a cluster meets thresholds for N consecutive ticks → promote to child Topic.
          7) Housekeeping: expire stale cluster states.

        Returns a summary dict (counts + promotions).
        """
        
        logger.debug(f"Starting tick for topic {topic_id}")
        now_ts = time.time()
        topic = self.storage.load_topic(topic_id)  # Durable state: long-term centroid, seeds, rules, etc.

        # -------- 1) Plan & harvest
        # Build concrete queries from Topic state (seeds + optional expansion near centroid).
        queries = self.planner.plan(topic, self.K_queries)
        logger.debug(f"Planned {len(queries)} queries: {queries}")

        # Execute searches. We intentionally fan out over queries to increase diversity of sources.
        hits: List[Dict] = []
        for q in queries:
            # Each hit should at minimum include a URL; adapters may add lightweight metadata.
            hits.extend(self.searcher.search(q, limit=10))

        # Fetch full pages (text + metadata) so we can embed.
        # NOTE: keep this I/O batch small enough to respect rate limits; you can parallelize in adapters.
        pages = [self.fetcher.fetch(h["url"]) for h in hits]

        # Convert texts → embeddings; encoder/version is abstracted behind the port.
        # The choice of encoder determines geometry but does not change the orchestration.
        vecs = self.embedder.embed([p["text"] for p in pages])

        # -------- 2) Build docs (authority/ts/domain/type are adapter-specific)
        # Dedup relies on a stable content hash; if adapter doesn't provide one, derive from text.
        seen = self.storage.seen(topic.id)  # Known content hashes for idempotent ingestion.
        new_docs: List[Doc] = []

        for p, v in zip(pages, vecs):
            # Fallback hash: deterministic hash of text; in production, prefer a strong content hash (e.g., SHA-256).
            h = p.get("hash") or str(hash(p["text"]))

            # Assemble the Doc. Authority, ts, and dtype are best-effort heuristics supplied by adapters.
            new_docs.append(
                Doc(
                    id=str(uuid.uuid4()),             # internal UUID, not the URL
                    ts=p.get("ts", now_ts),           # publish time if known, else now
                    url=p["url"],                     # canonical URL
                    domain=p.get("domain", ""),       # hostname for source-level rules
                    title=p.get("title", ""),         # human-readable
                    text=p["text"],                   # extracted plaintext
                    dtype=p.get("type", "unknown"),   # "paper" | "repo" | "blog" | "news" | ...
                    authority=p.get("authority", 0.5),# 0..1 trust/importance
                    vec=v,                            # embedding
                    hash=h,                           # for dedup
                    arm_id=p.get("arm_id", ""),       # retrieval arm/experiment tag (optional)
                    sample_weight=float(p.get("sample_weight", 1.0)),  # <-- carry through
                )
            )

        # -------- 3) Filter → Dedup → Rank(+MMR)
        # 3a) Negative rules: block by terms/domains/types before they pollute centroids.
        filtered = self.filter.apply(topic.negative, new_docs)

        # 3b) Dedup: drop already-seen content and trivial near-duplicates.
        unique = self.deduper.drop(seen, filtered)

        # 3c) Rank and diversify: pick a balanced slate of relevant, authoritative, and *non-redundant* docs.
        #     Typical implementation mixes cosine-to-centroid + authority + recency, then MMR for diversity.
        topK = self.ranker.select(topic, unique, K=self.K_keep)
        logger.debug(f"Filtered: {len(filtered)} → Unique: {len(unique)} → Top-K: {len(topK)}")

        # -------- 4) Update topic (long-term incremental mean; short-term EMA optional)
        # IMPORTANT: update long-term centroid using *new docs this tick* (not whole window).
        # This preserves the exact weighted mean over all history at O(1) per doc.
        self.updater.apply(topic, topK, now_ts)

        # Persist docs and update the "seen" set to keep ingestion idempotent.
        self.storage.save_docs(topic.id, topK)
        self.storage.mark_seen_hashes(topic.id, [d.hash for d in topK])

        # Persist updated Topic state (centroid_long, weight_sum, etc.) for crash-safety / horizontal scaling.
        self.storage.save_topic(topic)

        # -------- 5) Cluster recent window (structure detection uses window, not full history)
        # Pull a sliding window of recent docs (e.g., last 30 days) to reflect *current* structure.
        # The window size is a key knob: too small → noisy; too large → stale.
        window_docs = self.storage.recent_docs(topic.id, self.window_days, limit=500)

        # Cluster the window to get raw, per-tick snapshots (centroid_now, cohesion_now, separation_now, doc_ids).
        # NOTE: Cluster IDs from the algorithm are NOT stable across ticks (especially with HDBSCAN/k-means).
        snaps = self.clusterer.cluster(topic, window_docs)

        # -------- 6) Match raw clusters → persistent states, smooth metrics, promote if ready
        promotions: List[Topic] = []

        for snap in snaps:
            # (a) Find an existing ephemeral cluster by cosine(sim) to prior centroid_ema; else create a new state.
            #     This is CRITICAL: never rely on the raw algorithm's cluster_id across ticks.
            state = self.matcher.match_or_create(self.storage, topic, snap)

            # (b) Smooth metrics (EMA) & update persistence counter inside ClusterState.
            #     The smoother reads/writes persisted state and applies your β smoothing factor.
            state = self.smoother.update(self.storage, topic, snap, state)

            # (c) Check promotion rule on *smoothed* metrics + persistence.
            #     Using EMA avoids one-off spikes creating junk topics.
            if self.emergence.ready(topic, snap, state):
                # Materialize the cluster docs to generate a human-friendly name and seed terms.
                cluster_docs = [d for d in window_docs if d.id in snap.doc_ids]

                # Create a first-class child Topic with its own durable long-term centroid.
                child = self.emergence.promote(
                    parent=topic, snap=snap, namer=self.namer, cluster_docs=cluster_docs
                )
                self.storage.save_topic(child)
                topic.children.append(child)
                promotions.append(child)

                # Clean up ephemeral state to prevent re-promoting the same cluster on the next tick.
                self.storage.delete_cluster_state(topic.id, state.cluster_id)

        # Housekeeping: expire ephemeral clusters that weren't matched for a while.
        # The matcher may implement time/tick-based eviction; pass a monotonic counter if needed.
        self.matcher.expire_stale(self.storage, topic.id)

        # -------- Return a compact, machine-readable summary
        return {
            "ingested": len(topK),                        # docs accepted this tick
            "clusters_observed": len(snaps),              # raw clusters seen in window
            "promotions": [(t.id, t.name) for t in promotions],  # (id, name) for any new child topics
            "topic_id": topic.id,                         # echo input for downstream logging
            "updated_at": now_ts,                         # wall-clock timestamp of this run
        }
