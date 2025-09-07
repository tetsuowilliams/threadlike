import json, os, hashlib, time

class JsonCorpus:
    def __init__(self, root: str):
        self.root = root
        self.tick = 0

    def next_batch(self):
        path = os.path.join(self.root, f"tick_{self.tick:03}.json")
        self.tick += 1
        if not os.path.exists(path):
            return []  # end of corpus
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

def norm_text(s: str) -> str:
    # minimal normalization so small changes don't explode hashes
    return " ".join(s.lower().split())

def text_hash(s: str) -> str:
    return hashlib.sha256(norm_text(s).encode("utf-8")).hexdigest()

class FixtureSearch:
    """SearchPort impl: returns one 'hit' per entry (url only needed)."""
    def __init__(self, corpus: JsonCorpus): self.corpus = corpus
    def search(self, query: str, limit: int):
        # Ignore query—this fixture provides the exact batch for this tick.
        batch = self.corpus.next_batch()
        return [{"url": rec["url"]} for rec in batch]

class FixtureFetch:
    """FetchPort impl: returns full page objects from the same tick batch."""
    def __init__(self, corpus: JsonCorpus): 
        self.corpus = corpus
        self.cache = {}  # url -> last seen record (for idempotent fetch)

    def fetch(self, url: str):
        # In tests we preload the current tick's records into cache before calls,
        # or we provide a helper to map URL->record from the last next_batch().
        rec = self.cache.get(url)

        if rec is None:
            raise RuntimeError(f"URL {url} not preloaded into FixtureFetch.cache")

        # Ensure hash exists
        rec = dict(rec)
        rec["hash"] = rec.get("hash") or text_hash(rec["text"])
        return rec
