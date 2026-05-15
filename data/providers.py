"""
Market data providers — download and cache OHLCV data.

Uses yfinance by default (free, no API key). Polygon supported as optional.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd

logger = logging.getLogger("DataProvider")


def download_yfinance(
    symbols: List[str],
    start: str,
    end: str,
    cache_dir: Path,
) -> pd.DataFrame:
    """Download OHLCV data via yfinance. Caches to parquet."""
    import yfinance as yf

    cache_key = hashlib.md5(f"{sorted(symbols)}_{start}_{end}".encode()).hexdigest()[:12]
    cache_path = cache_dir / f"yfinance_{cache_key}.parquet"

    if cache_path.exists():
        logger.info(f"Loading cached data: {cache_path.name}")
        return pd.read_parquet(cache_path)

    logger.info(f"Downloading {len(symbols)} symbols from yfinance: {start} to {end}")
    cache_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for sym in symbols:
        try:
            sdf = yf.download(sym, start=start, end=end, auto_adjust=True, progress=False)
            # yfinance sometimes returns MultiIndex columns — flatten them
            if isinstance(sdf.columns, pd.MultiIndex):
                sdf.columns = sdf.columns.get_level_values(0)
            if len(sdf) > 0:
                sdf["Symbol"] = sym
                frames.append(sdf)
        except Exception as e:
            logger.warning(f"Failed to download {sym}: {e}")

    if not frames:
        raise RuntimeError("No data downloaded for any symbol.")
    df = pd.concat(frames)

    df.to_parquet(cache_path)
    logger.info(f"Cached {len(df)} rows to {cache_path.name}")
    return df


def load_data(
    symbols: List[str],
    start: str,
    end: str,
    source: str = "yfinance",
    cache_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Unified data loader."""
    if cache_dir is None:
        cache_dir = Path("data/cache")

    if source == "yfinance":
        return download_yfinance(symbols, start, end, cache_dir)
    else:
        raise ValueError(f"Unsupported data source: {source}. Use 'yfinance'.")
