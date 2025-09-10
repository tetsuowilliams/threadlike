from __future__ import annotations
from dataclasses import dataclass
from typing import List
from core_services.math_helpers import Vector

@dataclass
class Hit:
    url: str
    title: str
    domain: str
    type: str
    ts: float
    authority: float
    text: str
    embedding: Vector

@dataclass
class TopicHits:
    topic_id: str
    hits: List[Hit]