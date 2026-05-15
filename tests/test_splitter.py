"""Tests for backtest.splitter — embargo-aware temporal splits."""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import pytest
from backtest.splitter import TemporalDataSplitter


def _make_df(n: int) -> pd.DataFrame:
    rng = np.random.default_rng(99)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.DataFrame({"Close": close}, index=idx)


class TestTemporalDataSplitter:
    def test_default_split_sizes(self):
        df = _make_df(1000)
        s = TemporalDataSplitter()
        splits = s.split(df)
        assert set(splits.keys()) == {"train", "validation", "test"}
        assert len(splits["train"]) > 0
        assert len(splits["validation"]) > 0
        assert len(splits["test"]) > 0

    def test_embargo_creates_gap(self):
        df = _make_df(1000)
        s = TemporalDataSplitter(embargo_pct=0.02)
        splits = s.split(df)
        # train end index must be strictly before validation start index
        train_last = splits["train"].index[-1]
        val_first = splits["validation"].index[0]
        assert val_first > train_last

    def test_no_overlap_between_splits(self):
        df = _make_df(500)
        s = TemporalDataSplitter()
        splits = s.split(df)
        train_idx = set(splits["train"].index)
        val_idx = set(splits["validation"].index)
        test_idx = set(splits["test"].index)
        assert train_idx.isdisjoint(val_idx)
        assert train_idx.isdisjoint(test_idx)
        assert val_idx.isdisjoint(test_idx)

    def test_proportions_roughly_correct(self):
        n = 1000
        df = _make_df(n)
        s = TemporalDataSplitter(train_pct=0.6, val_pct=0.2, test_pct=0.2, embargo_pct=0.01)
        splits = s.split(df)
        # account for embargo rows
        total_data = len(splits["train"]) + len(splits["validation"]) + len(splits["test"])
        assert total_data < n  # some rows are embargoed
        assert len(splits["train"]) > len(splits["validation"])

    def test_raises_on_too_small_dataset(self):
        df = _make_df(3)  # way too small for 60/20/20 + embargo
        s = TemporalDataSplitter()
        with pytest.raises(ValueError, match="empty"):
            s.split(df)

    def test_temporal_order_preserved(self):
        df = _make_df(500)
        s = TemporalDataSplitter()
        splits = s.split(df)
        # train < val < test in chronological order
        assert splits["train"].index[-1] < splits["validation"].index[0]
        assert splits["validation"].index[-1] < splits["test"].index[0]
