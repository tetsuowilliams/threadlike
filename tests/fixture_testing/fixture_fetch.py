from protocols import Fetcher
from tests.fixture_testing.json_corpus import JsonCorpus
from tests.fixture_testing.text_helpers import text_hash


class FixtureFetch(Fetcher):
    """FetchPort impl: returns full page objects from the same tick batch."""
    def __init__(self, corpus: JsonCorpus): 
        self.corpus = corpus

    def fetch(self, url: str):
        # In tests we preload the current tick's records into cache before calls,
        # or we provide a helper to map URL->record from the last next_batch().
        cache = {rec["url"]: rec for rec in self.corpus.batch}

        rec = cache.get(url)

        if rec is None:
            raise RuntimeError(f"URL {url} not preloaded into FixtureFetch.cache")

        # Ensure hash exists
        rec = dict(rec)
        rec["hash"] = rec.get("hash") or text_hash(rec["text"])
        return rec
