# Set up logging first, before any other imports
from logging_config import setup_logging, get_logger
logger = setup_logging("INFO", "topic_evolver")

import uuid

from models import Topic, NegativeRules, TopicPolicy
from adapters.testing.in_memory_storage import InMemoryStorage
from adapters.testing.search.toy_query_planner import ToyQueryPlanner
from adapters.testing.search.toy_search import ToySearch
from adapters.testing.toy_fetch import ToyFetch
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
from orchestrator import TopicRefresher
from tests.fixture_testing.scenario import ScriptedScenario, build_corpus

def main():
    """Run the topic evolution simulation."""
    
    # Initialize storage
    storage = InMemoryStorage()
    
    # Create initial topic
    topic = Topic(
        id=str(uuid.uuid4()),
        name="New ML Models",
        seeds=["llm", "transformer", "benchmark"],
        negative=NegativeRules(),
        policy=TopicPolicy(
            m_min=6,
            tau_cohesion=0.55,
            tau_separation=0.70,
            persistence_min=2,
            ema_alpha_topic=0.10,
            ema_beta_cluster=0.25
        )
    )
    storage.save_topic(topic)
    logger.info(f"Created parent topic: {topic.name} ({topic.id})")
    
    # Setup scenario and corpus
    scenario = ScriptedScenario()
    corpus = build_corpus()
    logger.info(f"Built corpus with {len(corpus)} documents")
    
    # Wire up all components
    refresher = TopicRefresher(
        planner=ToyQueryPlanner(),
        search=ToySearch(scenario),
        fetcher=ToyFetch(corpus),
        embedder=ToyEmbed(dim=32),
        storage=storage,
        ranker=SimpleRanker(),
        filtr=PassFilter(),
        deduper=SeenDeduper(),
        clusterer=KMeans2Clusterer(),
        smoother=ClusterSmoother(beta=topic.policy.ema_beta_cluster),
        updater=TopicUpdater(recency_lambda=0.00),
        emergence=EmergenceDetector(),
        namer=SimpleNamer(),
        matcher=ClusterMatcher(),
        window_days=30,
        K_queries=6,
        K_keep=20
    )
    logger.info("Initialized all components")
    
    # Run simulation
    logger.info(f"Starting simulation for topic: {topic.name}")
    for tick in range(8):
        logger.info(f"Running tick {tick}...")
        out = refresher.tick(topic.id)
        
        # Color-coded output based on results
        if out['promotions']:
            logger.warning(f"Tick {tick}: ingested={out['ingested']}, clusters={out['clusters_observed']}, promotions={out['promotions']}")
        else:
            logger.info(f"Tick {tick}: ingested={out['ingested']}, clusters={out['clusters_observed']}, promotions={out['promotions']}")
    
    # Show any child topics created
    children = [t for t in storage.topics.values() if t.emerged_from == topic.id]
    if children:
        logger.warning(f"Found {len(children)} promoted topics:")
        for c in children:
            logger.warning(f"  🎉 PROMOTED → {c.name} ({c.id})")
    else:
        logger.info("No topics were promoted during simulation")


if __name__ == "__main__":
    main()