from protocols import Storage
from models import Topic

from dataclasses import dataclass
from typing import List, Optional
from core_services.math_helpers import Vector


@dataclass
class ClusterObservation:
    id: str
    centroid_ema: Optional[Vector] = None
    cohesion_ema: float = 0.0
    separation_ema: float = 0.0
    persistence: int = 0

@dataclass
class TopicObservation:
    topic_id: str
    seeds: List[str]
    clusters: List[ClusterObservation]
    centroid_long: Optional[Vector] = None
    weight_sum: float = 0.0
    centroid_short_ema: Optional[Vector] = None

@dataclass
class Observation:
    tick: int
    topic: Topic
    

class Observer:
    def __init__(self, storage: Storage):
        self.storage = storage
        self.tick = 0
        self.observations: List[Observation] = []

    def observe_on_tick(self):
        topics = self.storage.get_all_topics()

        for topic in topics:
            topic_observation = TopicObservation(
                topic_id=topic.id, 
                seeds=topic.seeds, 
                centroid_long=topic.centroid_long, 
                weight_sum=topic.weight_sum, 
                centroid_short_ema=topic.centroid_short_ema,
                clusters=[]
            )

            for cluster in self.storage.get_all_cluster_states_for_topic(topic.id):
                cluster_observation = ClusterObservation(
                    id=cluster.cluster_id, 
                    centroid_ema=cluster.centroid_ema,
                    cohesion_ema=cluster.cohesion_ema,
                    separation_ema=cluster.separation_ema,
                    persistence=cluster.persistence
                )
                topic_observation.clusters.append(cluster_observation)

            self.observations.append(
                Observation(
                    tick=self.tick, 
                    topic=topic_observation
                )
            )

        self.tick += 1
        return self.observations