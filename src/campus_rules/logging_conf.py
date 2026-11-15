"""轻量日志：只打印能力状态与链路耗时，不打印密钥、不打印正文全文。"""
from __future__ import annotations

import logging
import sys

_CONFIGURED = False


def setup_logging(level: str = "INFO") -> logging.Logger:
    global _CONFIGURED
    logger = logging.getLogger("campus_rules")
    if not _CONFIGURED:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)-7s | %(name)s.%(funcName)s | %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        logger.addHandler(handler)
        logger.propagate = False
        _CONFIGURED = True
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    return logger


def get_logger(name: str = "pipeline") -> logging.Logger:
    return logging.getLogger(f"campus_rules.{name}")
