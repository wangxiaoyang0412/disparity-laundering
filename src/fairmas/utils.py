"""Shared utilities: reproducible seeding and logging setup."""

from __future__ import annotations

import logging
import os
import random


def set_seed(seed: int = 42) -> None:
    """Seed Python and (if available) NumPy for reproducibility.

    Torch is intentionally not imported here — serving runs in a separate env and
    the client side only needs deterministic sampling of cases/order.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np  # local import keeps the base client torch/numpy-free

        np.random.seed(seed)
    except ImportError:
        pass


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a module logger with a single stream handler (idempotent)."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger


__all__ = ["set_seed", "get_logger"]
