import logging
from pathlib import Path

# Toggle: False -> colored console (rich), True -> plain text file.
LOG_TO_FILE = False

LOG_LEVEL = logging.INFO
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# File handler keeps the full, greppable layout. The console handler (rich) draws
# its own time/level/path columns, so it only needs the logger name + message.
FILE_FORMAT = "%(asctime)s [%(levelname)s] %(name)s %(filename)s:%(lineno)d - %(message)s"
CONSOLE_FORMAT = "%(name)s - %(message)s"

# <project root>/logs/ssnanopore_rag.log
LOG_FILE = Path(__file__).resolve().parents[3] / "logs" / "ssnanopore_rag.log"


def setup_logging(level: int = LOG_LEVEL) -> None:
    if LOG_TO_FILE:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        handler: logging.Handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
        handler.setFormatter(logging.Formatter(FILE_FORMAT, datefmt=DATE_FORMAT))
    else:
        from rich.logging import RichHandler

        handler = RichHandler(
            rich_tracebacks=True,
            tracebacks_show_locals=True,
            markup=True,
            show_path=True,
            log_time_format=f"[{DATE_FORMAT}]",
        )
        handler.setFormatter(logging.Formatter(CONSOLE_FORMAT))

    app_logger = logging.getLogger("ssnanopore_rag")
    app_logger.setLevel(level)
    app_logger.handlers.clear()
    app_logger.addHandler(handler)
    app_logger.propagate = False

    # Disable logging for all imported packages
    root = logging.getLogger()
    root.setLevel(logging.CRITICAL)
    root.handlers.clear()
    root.addHandler(logging.NullHandler())
