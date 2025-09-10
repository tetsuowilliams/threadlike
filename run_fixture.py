from logging_config import setup_logging, get_logger
logger = setup_logging("DEBUG", "run_fixture")


# Fixture I/O
from tests.fixture_testing.json_corpus import JsonCorpus
from tests.fixture_testing.fixture_test import FixtureTest


def main():
    logger.info("================================================================================================")
    logger.info("================================================================================================")
    logger.info("Starting test bench loop simulation")
    
    # --- Fixture corpus & fetcher
    corpus = JsonCorpus("tests/fixture_testing/corpus/test1")
    logger.info("Loaded fixture corpus and fetcher")

    fixture_test = FixtureTest(corpus, 10)
    fixture_test.test()

if __name__ == "__main__":
    main()
