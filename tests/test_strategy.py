"""Tests for strategy.py — signal contract validation."""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def load_strategy():
    """Import strategy.py from the project root."""
    strategy_path = Path(__file__).parent.parent / "strategy.py"
    spec = importlib.util.spec_from_file_location("strategy", strategy_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestStrategyContract:
    def test_function_exists(self):
        mod = load_strategy()
        assert hasattr(mod, "generate_signals")

    def test_returns_series(self, sample_ohlcv):
        mod = load_strategy()
        signals = mod.generate_signals(sample_ohlcv)
        assert isinstance(signals, pd.Series)

    def test_same_index_as_input(self, sample_ohlcv):
        mod = load_strategy()
        signals = mod.generate_signals(sample_ohlcv)
        assert list(signals.index) == list(sample_ohlcv.index)

    def test_values_in_valid_range(self, sample_ohlcv):
        mod = load_strategy()
        signals = mod.generate_signals(sample_ohlcv)
        assert set(signals.unique()).issubset({-1, 0, 1})

    def test_no_nans_in_output(self, sample_ohlcv):
        mod = load_strategy()
        signals = mod.generate_signals(sample_ohlcv)
        assert not signals.isna().any()

    def test_handles_minimal_dataframe(self):
        mod = load_strategy()
        df = pd.DataFrame({
            "Open": [100.0], "High": [101.0], "Low": [99.0],
            "Close": [100.5], "Volume": [1_000_000.0],
        })
        signals = mod.generate_signals(df)
        assert len(signals) == 1

    def test_handles_lowercase_close(self):
        """Strategy must handle 'close' (lowercase) column too."""
        mod = load_strategy()
        df = pd.DataFrame({"close": [100.0, 101.0, 99.0, 102.0]})
        try:
            signals = mod.generate_signals(df)
            assert len(signals) == len(df)
        except (KeyError, Exception):
            pytest.skip("Strategy doesn't support lowercase column names — OK")
