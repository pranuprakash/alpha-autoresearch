"""Embargo-aware temporal data splitting."""

from __future__ import annotations

import logging
from typing import Dict

import pandas as pd

logger = logging.getLogger("Splitter")


class TemporalDataSplitter:
    """
    Prevents information leakage between train/validation/test sets by
    inserting an embargo gap between each segment.
    """

    def __init__(
        self,
        train_pct: float = 0.60,
        val_pct: float = 0.20,
        test_pct: float = 0.20,
        embargo_pct: float = 0.02,
    ):
        self.train_pct = train_pct
        self.val_pct = val_pct
        self.test_pct = test_pct
        self.embargo_pct = embargo_pct

    def split(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        n = len(df)
        embargo_rows = max(int(n * self.embargo_pct), 1)

        train_end = int(n * self.train_pct)
        val_start = train_end + embargo_rows
        val_end = val_start + int(n * self.val_pct)
        test_start = val_end + embargo_rows

        splits = {
            "train": df.iloc[:train_end].copy(),
            "validation": df.iloc[val_start:val_end].copy(),
            "test": df.iloc[test_start:].copy(),
        }

        for name, split_df in splits.items():
            if len(split_df) == 0:
                raise ValueError(
                    f"Data split '{name}' is empty. Dataset too small "
                    f"for the configured split ratios (n={n})."
                )

        logger.info(
            f"Data split: train={len(splits['train'])}, "
            f"val={len(splits['validation'])}, test={len(splits['test'])}, "
            f"embargo={embargo_rows} rows"
        )
        return splits

    def get_split_metadata(self, df: pd.DataFrame, date_col: str = "Date") -> Dict:
        splits = self.split(df)
        meta = {}
        for name, split_df in splits.items():
            meta[name] = {
                "rows": len(split_df),
                "start": str(split_df[date_col].iloc[0]) if date_col in split_df.columns else str(split_df.index[0]),
                "end": str(split_df[date_col].iloc[-1]) if date_col in split_df.columns else str(split_df.index[-1]),
            }
        return meta
