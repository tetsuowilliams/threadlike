import uuid
from typing import List, Dict

from logging_config import get_logger

from tests.fixture_testing.json_corpus import JsonCorpus
from tests.fixture_testing.fixture_fetch import FixtureFetch
from tests.fixture_testing.fixture_search import FixtureSearch
from adapters.testing.in_memory_storage import InMemoryStorage
from models import Topic, NegativeRules, TopicPolicy
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


class FixtureTest:
    def __init__(self, corpus: JsonCorpus, num_ticks: int):
        self.corpus = corpus
        self.logger = get_logger("FixtureTest")
        self.num_ticks = num_ticks

    def print_topic_tree(self, topic: Topic, storage: InMemoryStorage, depth: int = 0):
        """Print topic tree with --- indentation."""
        indent = "---" * depth
        self.logger.info(f"{indent}Topic {topic.id}: {topic.name}")
        
        # Get child topics (if any)
        child_topics = [t for t in storage.topics.values() if t.emerged_from == topic.id]
        for child in child_topics:
            self.print_topic_tree(child, storage, depth + 1)

    def print_cluster_states(self, topic: Topic, storage: InMemoryStorage):
        self.logger.info(f"Cluster states for topic {topic.id}: {len(storage.get_all_cluster_states_for_topic(topic.id))}")
        for cluster_state in storage.get_all_cluster_states_for_topic(topic.id):
            self.logger.info(f"  Cluster state {cluster_state.cluster_id}")

    def test(self):
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
        self.logger.info(f"Created topic: {topic.name} ({topic.id})")

        # --- Fixture corpus & fetcher
        fetch = FixtureFetch(self.corpus)
        search = FixtureSearch(self.corpus)
        self.logger.info("Loaded fixture corpus and fetcher")    
        
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
            observer=Observer(storage),
            window_days=30,
            K_queries=6,
            K_keep=20,
        )

        for _ in range(self.num_ticks):
            orchestrator.tick(topic.id)

            if not self.corpus.next_batch():
                break
        
        self.logger.info("Test bench loop simulation completed")

        no_topics = len(storage.topics.values())
        self.logger.info(f"Number of topics: {no_topics}")
        self.logger.info("Printing topic tree and cluster states")

        # Print only root topics (those without emerged_from)
        root_topics = [t for t in storage.topics.values() if t.emerged_from is None]
        for topic in root_topics:
            self.print_topic_tree(topic, storage)
            self.print_cluster_states(topic, storage)
