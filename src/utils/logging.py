import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

_GLOBAL_LOG_FILE: Optional[str] = None


def get_logger(
    name: str = "preprocessing", log_file: Optional[str] = None
) -> logging.Logger:
    """
    Creates a logger with both console and file handlers.
    If log_file is not provided, it uses a timestamped default if not already set.
    """
    global _GLOBAL_LOG_FILE

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # If handlers are already set, don't add them again
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler
    if log_file:
        _GLOBAL_LOG_FILE = log_file

    if _GLOBAL_LOG_FILE is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        _GLOBAL_LOG_FILE = f"logs/preprocessing-{timestamp}.log"

    log_path = Path(_GLOBAL_LOG_FILE)
    if not log_path.parent.exists():
        log_path.parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
