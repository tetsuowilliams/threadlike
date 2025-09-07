# Set up logging first, before any other imports
from logging_config import setup_logging, get_logger
logger = setup_logging("DEBUG", "test_bench_loop")

import uuid
from typing import List, Dict

from models import Topic, NegativeRules, TopicPolicy
from adapters.testing.in_memory_storage import InMemoryStorage
from adapters.testing.search.toy_query_planner import ToyQueryPlanner
from adapters.testing.toy_embed import ToyEmbed
from adapters.testing.filtering.pass_filter import PassFilter
from adapters.testing.filtering.seen_deduper import SeenDeduper
from adapters.testing.filtering.simple_ranker import SimpleRanker
from adapters.testing.kmeans2_clusterer import KMeans2Clusterer
from adapters.testing.simple_namer import SimpleNamer

from core_services.topic_updater import TopicUpdater
from core_services.cluster_smoother import ClusterSmoother
from core_services.emergence_detector import EmergenceDetector
from core_services.cluster_matcher import ClusterMatcher
from core_services.orchestrator import Orchestrator

# Fixture I/O
from tests.fixture_testing.fixture_reader import JsonCorpus, FixtureFetch

NUM_TICKS = 10  # or let the loop break when corpus ends

class FixtureSearchAdapter:
    """Minimal SearchPort that serves the URLs for the current preloaded batch."""
    def __init__(self, current_batch_getter):
        self._get = current_batch_getter

    def search(self, query: str, limit: int) -> List[Dict]:
        # Ignore query; return one hit per record in the current tick.
        return [{"url": r["url"]} for r in self._get()][:limit]




def print_topic_tree(topic: Topic, storage: InMemoryStorage, depth: int = 0):
    """Print topic tree with --- indentation."""
    indent = "---" * depth
    logger.info(f"{indent}Topic {topic.id}: {topic.name}")
    
    # Get child topics (if any)
    child_topics = [t for t in storage.topics.values() if t.emerged_from == topic.id]
    for child in child_topics:
        print_topic_tree(child, storage, depth + 1)

def print_cluster_states(topic: Topic, storage: InMemoryStorage):
    logger.info(f"Cluster states for topic {topic.id}: {len(storage.get_all_cluster_states_for_topic(topic.id))}")
    for cluster_state in storage.get_all_cluster_states_for_topic(topic.id):
        logger.info(f"  Cluster state {cluster_state.cluster_id}")

def main():
    logger.info("================================================================================================")
    logger.info("================================================================================================")
    logger.info("Starting test bench loop simulation")
    
    # --- Storage + initial topic
    storage = InMemoryStorage()
    topic = Topic(
        id=str(uuid.uuid4()),
        name="New ML Models",
        seeds=["llm", "transformer", "benchmark"],
        negative=NegativeRules(),
        policy=TopicPolicy(
            m_min=3,                 # lower mass threshold for tests
            tau_cohesion=0.55,       # reasonable tightness
            tau_separation=0.80,     # allow cos(parent,cluster) ≤ 0.80
            persistence_min=1,       # prove the pipe can promote
            ema_alpha_topic=0.10,
            ema_beta_cluster=0.40,
        ),
    )
    storage.save_topic(topic)
    logger.info(f"Created topic: {topic.name} ({topic.id})")

    # --- Fixture corpus & fetcher
    corpus = JsonCorpus("tests/fixture_testing/corpus/test1")
    fetch = FixtureFetch(corpus)
    logger.info("Loaded fixture corpus and fetcher")

    # This list will hold the current tick's (possibly deduped) records.
    current_batch: List[Dict] = []

    def get_current_batch() -> List[Dict]:
        return current_batch
    
    matcher = ClusterMatcher()

    # --- Wire orchestrator with fixture adapters
    refresher = Orchestrator(
        planner=ToyQueryPlanner(),
        searcher=FixtureSearchAdapter(get_current_batch),  # <- fixture search
        fetcher=fetch,                                   # <- fixture fetch
        embedder=ToyEmbed(dim=32),
        storage=storage,
        ranker=SimpleRanker(),
        filtr=PassFilter(),
        deduper=SeenDeduper(),           # within-tick dedup handled below (pre-Doc)
        clusterer=KMeans2Clusterer(),
        smoother=ClusterSmoother(beta=topic.policy.ema_beta_cluster),
        updater=TopicUpdater(recency_lambda=0.00),
        emergence=EmergenceDetector(),
        namer=SimpleNamer(),
        matcher=matcher,
        window_days=30,
        K_queries=6,
        K_keep=20,
    )

    # --- Drive ticks from on-disk JSON
    logger.info(f"Starting {NUM_TICKS} ticks from fixture data")
    for tick_num in range(NUM_TICKS):
        # Load the next tick's raw records and prime the fetch cache
        current_batch = corpus.next_batch()

        if not current_batch:
            logger.info("No more tick files available, ending simulation")
            break  # no more tick files

        fetch.cache = {rec["url"]: rec for rec in current_batch}
        logger.debug(f"Loaded {len(current_batch)} records for tick {tick_num}")

        # Run one orchestrator tick
        out = refresher.tick(topic.id)

        clusters = matcher.list_states(storage, topic.id)
        cluster_length = len(clusters)
        logger.info(f"Clusters for topic {topic.id}: {cluster_length}")

        # Use colorful logging instead of print
        if out['promotions']:
            logger.warning(f"Tick {tick_num}: ingested={out['ingested']}, clusters={out['clusters_observed']}, promotions={out['promotions']}")
        else:
            logger.info(f"Tick {tick_num}: ingested={out['ingested']}, clusters={out['clusters_observed']}, promotions={out['promotions']}")
    
    logger.info("Test bench loop simulation completed")

    no_topics = len(storage.topics.values())
    logger.info(f"Number of topics: {no_topics}")
    logger.info("Printing topic tree and cluster states")

    # Print only root topics (those without emerged_from)
    root_topics = [t for t in storage.topics.values() if t.emerged_from is None]
    for topic in root_topics:
        print_topic_tree(topic, storage)
        print_cluster_states(topic, storage)

if __name__ == "__main__":
    main()
