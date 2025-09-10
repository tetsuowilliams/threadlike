import os
import json


class JsonCorpus:
    def __init__(self, root: str):
        self.root = root
        self.tick = 0
        self.batch = []
        self.next_batch()

    def next_batch(self) -> bool:
        path = os.path.join(self.root, f"tick_{self.tick:03}.json")
        self.tick += 1
        
        if not os.path.exists(path):
            return False  # end of corpus

        with open(path, "r", encoding="utf-8") as f:
            self.batch = json.load(f)
            return True
