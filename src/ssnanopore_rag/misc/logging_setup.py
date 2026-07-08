import logging
from pathlib import Path

# Toggle: False prints to console, True writes to a file.
LOG_TO_FILE = False

LOG_LEVEL = logging.INFO
# time | level | logger | file:line | message
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(filename)s:%(lineno)d | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# <project root>/logs/ssnanopore_rag.log
LOG_FILE = Path(__file__).resolve().parents[3] / "logs" / "ssnanopore_rag.log"


def setup_logging(level: int = LOG_LEVEL) -> None:
    if LOG_TO_FILE:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        handler: logging.Handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    else:
        handler = logging.StreamHandler()

    handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(handler)
