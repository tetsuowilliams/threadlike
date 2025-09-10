from logging_config import get_logger

from tests.fixture_testing.json_corpus import JsonCorpus
from adapters.testing.in_memory_storage import InMemoryStorage
from core_services.orchestrator import Orchestrator
from observer import Observer
from models import Topic


class FixtureTest:
    def __init__(self, 
        corpus: JsonCorpus, 
        observer: Observer, 
        orchestrator: Orchestrator, 
        topic: Topic,
        num_ticks: int):
        self.corpus = corpus
        self.topic = topic
        self.logger = get_logger("FixtureTest")
        self.num_ticks = num_ticks
        self.observer = observer
        self.orchestrator = orchestrator

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
        for _ in range(self.num_ticks):
            self.orchestrator.tick(self.topic.id)
            self.observer.observe_on_tick()

            if not self.corpus.next_batch():
                break
        
        self.logger.info("Test bench loop simulation completed")
