"""Logging."""

import logging
import os
import sys
from pathlib import Path

from sta_baseline.utils import distributed


def setup_logging(output_dir: str | None = None) -> None:
    """Sets up the logging for multiple processes.

    Only enable the logging for the master process, and suppress logging for the non-master processes.
    """
    logger_lightning = logging.getLogger("lightning")
    logger_lightning.handlers = []

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    plain_formatter = logging.Formatter(
        "[%(asctime)s][%(levelname)s] %(name)s: %(lineno)4d: %(message)s",
        datefmt="%m/%d %H:%M:%S",
    )
    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(plain_formatter)
        logger.addHandler(ch)

    if output_dir is not None:
        filename = Path(output_dir) / f"stdout_{distributed.get_rank()}.log"
        fh = logging.FileHandler(filename)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(plain_formatter)
        logger.addHandler(fh)


def get_logger(name: str | None = None) -> logging.Logger:
    """Retrieve the logger with the specified name.

    If name is None, return a logger which is the root logger of the hierarchy.

    Args:
        name (string): name of the logger.

    Returns:
        logging.Logger: the logger with the specified name.
    """
    return logging.getLogger(name)
