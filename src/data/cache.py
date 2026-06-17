"""Parquet-backed cache so backtests run offline after one WRDS pull."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pandas as pd

from ..config import REPO_ROOT


def cache_dir() -> Path:
    d = Path(os.environ.get("CB_CACHE_DIR", REPO_ROOT / "data_cache"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _key(name: str, *parts: str) -> str:
    h = hashlib.sha1("|".join(parts).encode()).hexdigest()[:10]
    return f"{name}_{h}.parquet"


def load(name: str, *parts: str) -> pd.DataFrame | None:
    path = cache_dir() / _key(name, *parts)
    if path.exists():
        return pd.read_parquet(path)
    return None


def save(df: pd.DataFrame, name: str, *parts: str) -> Path:
    path = cache_dir() / _key(name, *parts)
    df.to_parquet(path)
    return path
