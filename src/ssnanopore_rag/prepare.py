import logging

import dotenv

dotenv.load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    logger.info("Prepare data to embed and store...")


if __name__ == "__main__":
    main()
