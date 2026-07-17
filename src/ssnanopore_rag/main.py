import logging

import dotenv

from ssnanopore_rag import prepare, run
from ssnanopore_rag.misc.logging_setup import setup_logging

dotenv.load_dotenv()
logger = logging.getLogger(__name__)
setup_logging()
logger.info("Starting SSNanopore RAG pipeline...")


def main():
    prepare.main()
    run.main()

    logger.info("Pipeline finished.")


if __name__ == "__main__":
    main()
