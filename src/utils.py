"""Shared helpers: config loading, seeding, logging, path resolution."""
from __future__ import annotations

import logging
import os
import random
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path = ROOT / "config.yaml") -> dict:
    """Load the YAML configuration file into a dict."""
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve(rel_path: str | Path) -> Path:
    """Resolve a repo-relative path to an absolute one."""
    return ROOT / rel_path


def set_seed(seed: int = 42) -> None:
    """Seed every RNG we use so reruns are deterministic."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a consistently formatted logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
        )
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def split_boundaries(cfg: dict) -> dict:
    """Chronological train / validation / test cut-offs derived from the config.

    Whole days, earliest -> train, middle -> validation, latest -> test. Anything that is
    *fitted* (imputation medians, scalers) must use only timestamps < ``train_end``.
    """
    import pandas as pd

    start = pd.Timestamp(cfg["data"]["date_start"])
    n_days = (pd.Timestamp(cfg["data"]["date_end"]) - start).days + 1
    n_train = int(n_days * cfg["split"]["train_frac"])
    n_val = int(n_days * cfg["split"]["val_frac"])
    return {
        "train_end": start + pd.Timedelta(days=n_train),
        "val_end": start + pd.Timedelta(days=n_train + n_val),
        "n_train_days": n_train, "n_val_days": n_val, "n_test_days": n_days - n_train - n_val,
    }
