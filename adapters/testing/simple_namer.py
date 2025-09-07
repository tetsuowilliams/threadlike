from typing import List, Dict, Tuple
from models import Doc
from protocols import EmergenceNamer

from logging_config import get_logger
logger = get_logger(__name__)


class SimpleNamer(EmergenceNamer):
    """Simple naming based on frequent words in cluster documents."""
    
    def name_and_seeds(self, cluster_docs: List[Doc]) -> Tuple[str, List[str]]:
        text = " ".join(d.text for d in cluster_docs)
        words = text.lower().split()

        # top 3 frequent non-stopword tokens (toy)
        stops = {"the", "and", "of", "a", "to", "in", "on", "for", "with", "is", "are"}
        freq: Dict[str, int] = {}
        
        for w in words:
            if w in stops or len(w) < 3:
                continue
            freq[w] = freq.get(w, 0) + 1
        
        seeds = [w for w, _ in sorted(freq.items(), key=lambda kv: kv[1], reverse=True)[:3]]
        name = "Topic: " + (", ".join(seeds) if seeds else "emergent")
        return name, seeds or ["emergent"]
