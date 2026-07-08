"""
Entry point: configures logging once, then runs the prepare and run stages.

Because logging is set up here at the root logger, every module that does
``logger = logging.getLogger(__name__)`` (e.g. prepare, run) inherits this
configuration automatically.
"""

import logging

from ssnanopore_rag import prepare, run
from ssnanopore_rag.misc.logging_setup import setup_logging

logger = logging.getLogger(__name__)


def main() -> None:
    setup_logging()
    logger.info("Starting SSNanopore RAG pipeline...")

    prepare.main()
    run.main()

    logger.info("Pipeline finished.")


if __name__ == "__main__":
    main()
