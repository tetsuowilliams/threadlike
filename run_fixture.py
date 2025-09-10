from logging_config import setup_logging, get_logger
logger = setup_logging("DEBUG", "run_fixture")


# Fixture I/O
from tests.fixture_testing.json_corpus import JsonCorpus
from tests.fixture_testing.fixture_test import FixtureTest
from adapters.testing.in_memory_storage import InMemoryStorage
from models import Topic, NegativeRules, TopicPolicy
from observer import Observer
from tests.fixture_testing.fixture_fetch import FixtureFetch
from tests.fixture_testing.fixture_search import FixtureSearch
from adapters.testing.in_memory_storage import InMemoryStorage
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
from observer import Observer

import uuid

def main():
    logger.info("================================================================================================")
    logger.info("================================================================================================")
    logger.info("Starting test bench loop simulation")
    
    # --- Fixture corpus & fetcher
    corpus = JsonCorpus("tests/fixture_testing/corpus/test1")
    logger.info("Loaded fixture corpus and fetcher")

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
    fetch = FixtureFetch(corpus)
    search = FixtureSearch(corpus)
    logger.info("Loaded fixture corpus and fetcher")    
    
    matcher = ClusterMatcher()

    # --- Wire orchestrator with fixture adapters
    orchestrator = Orchestrator(
        planner=ToyQueryPlanner(),
        searcher=search,  # <- fixture search
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

    fixture_test = FixtureTest(corpus, Observer(storage), orchestrator, topic, 10)
    fixture_test.test()

if __name__ == "__main__":
    main()
