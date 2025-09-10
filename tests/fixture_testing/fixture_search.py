from protocols.searcher import Searcher
from tests.fixture_testing.json_corpus import JsonCorpus


class FixtureSearch(Searcher):
    def __init__(self, corpus: JsonCorpus): 
        self.corpus = corpus
    
    def search(self, query: str, limit: int):
        # Ignore query—this fixture provides the exact batch for this tick.
        return [{"url": rec["url"]} for rec in self.corpus.batch]
