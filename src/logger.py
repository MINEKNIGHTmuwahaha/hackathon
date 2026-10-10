"""
src/logger.py
-------------
Unified logger with console + optional file output.
"""

import logging
import sys

from pathlib import Path


_LOGGERS: dict[str, logging.Logger] = {}


def get_logger(name: str = "sentinel",
               level: str = "INFO",
               log_file: str | None = None) -> logging.Logger:
    if name in _LOGGERS:
        return _LOGGERS[name]

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    if not logger.handlers:
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )

        # Console
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        logger.addHandler(sh)

        # File
        if log_file:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setFormatter(fmt)
            logger.addHandler(fh)

    _LOGGERS[name] = logger
    return logger