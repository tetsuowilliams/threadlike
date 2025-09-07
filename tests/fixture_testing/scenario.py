"""Scenario and corpus building for testing the topic evolution system."""

from __future__ import annotations
from typing import List, Dict
import time


class ScriptedScenario:
    """Feeds URL keys per tick. First ticks are general LLM; later ticks emphasize 'reasoning/CoT/R1'."""
    
    def __init__(self):
        self.t = 0
        self.schedule: List[List[str]] = []
        # Build 8 ticks; last 4 emphasize new vocab
        for i in range(4):
            self.schedule.append([f"url_llm_{i}_{j}" for j in range(5)])  # 5 general docs
        for i in range(4, 8):
            # Mix: 2 general + 6 reasoning docs
            self.schedule.append([f"url_llm_{i}_{j}" for j in range(2)] + [f"url_reason_{i}_{j}" for j in range(6)])
    
    def pop_batch(self) -> List[str]:
        """Get the next batch of URLs for the current tick."""
        if self.t >= len(self.schedule):
            return []
        b = self.schedule[self.t]
        self.t += 1
        return b


def build_corpus() -> Dict[str, Dict]:
    """Build a test corpus with general LLM and reasoning-focused documents."""
    corpus = {}
    now = time.time()
    
    def mk(url: str, txt: str, auth: float = 0.6) -> Dict:
        return {
            "url": url,
            "text": txt,
            "title": url,
            "ts": now + hash(url) % 1000,
            "authority": auth
        }
    
    # General LLM docs
    for i in range(8):
        for j in range(6):
            url = f"url_llm_{i}_{j}"
            txt = "LLM transformer attention benchmark model release architecture training dataset"
            corpus[url] = mk(url, txt, auth=0.6)
    
    # Reasoning cluster docs (emergent vocabulary)
    for i in range(8):
        for j in range(8):
            url = f"url_reason_{i}_{j}"
            txt = "reasoning chain-of-thought deliberate R1 evaluation math proofs deep thinking self-improve"
            corpus[url] = mk(url, txt, auth=0.8)
    
    return corpus
