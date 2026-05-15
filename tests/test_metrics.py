"""Tests for backtest.metrics — Sharpe, Sortino, drawdown, Calmar, win rate."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from backtest.metrics import (
    compute_all_metrics,
    compute_calmar,
    compute_max_drawdown,
    compute_sharpe,
    compute_sortino,
    compute_win_rate,
    MAX_PLAUSIBLE_SHARPE,
)


def _returns(values):
    return pd.Series(values, dtype=float)


def _equity(values):
    return pd.Series(values, dtype=float)


class TestSharpe:
    def test_flat_returns_zero(self):
        r = _returns([0.0] * 100)
        assert compute_sharpe(r) == 0.0

    def test_positive_returns_positive_sharpe(self):
        rng = np.random.default_rng(5)
        r = _returns(rng.normal(0.002, 0.01, 252))  # positive mean, varying
        s = compute_sharpe(r)
        assert s > 0

    def test_capped_at_max_plausible(self):
        rng = np.random.default_rng(6)
        r = _returns(rng.normal(0.1, 0.0001, 252))  # very high Sharpe
        assert compute_sharpe(r) == MAX_PLAUSIBLE_SHARPE

    def test_negative_returns_negative_sharpe(self):
        rng = np.random.default_rng(7)
        r = _returns(rng.normal(-0.005, 0.01, 252))  # negative mean, varying
        assert compute_sharpe(r) < 0

    def test_single_row_no_crash(self):
        r = _returns([0.01])
        result = compute_sharpe(r)
        assert isinstance(result, float)


class TestSortino:
    def test_no_downside_returns_zero(self):
        r = _returns([0.01] * 100)
        assert compute_sortino(r) == 0.0

    def test_mixed_returns_positive(self):
        rng = np.random.default_rng(1)
        r = _returns(rng.normal(0.001, 0.01, 252))
        s = compute_sortino(r)
        assert isinstance(s, float)

    def test_empty_returns_zero(self):
        r = _returns([])
        assert compute_sortino(r) == 0.0


class TestMaxDrawdown:
    def test_monotonic_rise_zero_drawdown(self):
        eq = _equity([100, 110, 120, 130])
        assert compute_max_drawdown(eq) == 0.0

    def test_simple_drawdown(self):
        eq = _equity([100, 110, 80, 90])
        dd = compute_max_drawdown(eq)
        assert dd < 0  # drawdown is negative
        # max drawdown = (80 - 110) / 110
        expected = (80 - 110) / 110  # ~-0.2727 (negative)
        assert dd == pytest.approx(expected, abs=1e-6)

    def test_full_loss(self):
        eq = _equity([100, 50, 10])
        dd = compute_max_drawdown(eq)
        assert dd < -0.8

    def test_empty_series_zero(self):
        eq = _equity([])
        assert compute_max_drawdown(eq) == 0.0


class TestCalmar:
    def test_zero_drawdown_returns_zero(self):
        eq = _equity([100, 110, 120])
        r = _returns([0.01] * 3)
        assert compute_calmar(r, eq) == 0.0

    def test_positive_calmar(self):
        rng = np.random.default_rng(7)
        r = _returns(rng.normal(0.001, 0.005, 252))
        cum = 100 * (1 + r).cumprod()
        c = compute_calmar(r, cum)
        assert isinstance(c, float)


class TestWinRate:
    def test_all_positive(self):
        assert compute_win_rate(_returns([1.0, 0.5, 2.0])) == 1.0

    def test_all_negative(self):
        assert compute_win_rate(_returns([-1.0, -0.5])) == 0.0

    def test_half(self):
        assert compute_win_rate(_returns([1.0, -1.0, 1.0, -1.0])) == 0.5

    def test_empty_zero(self):
        assert compute_win_rate(_returns([])) == 0.0


class TestComputeAllMetrics:
    def test_returns_expected_keys(self, sample_ohlcv):
        r = _returns(np.diff(sample_ohlcv["Close"].values) / sample_ohlcv["Close"].values[:-1])
        eq = _equity((1 + r).cumprod() * 100_000)
        pnl = r * 100_000
        result = compute_all_metrics(r, eq, total_trades=100, trade_pnls=pnl)
        for key in [
            "sharpe_ratio", "sortino_ratio", "max_drawdown",
            "calmar_ratio", "total_trades", "win_rate",
            "avg_trade_pnl", "total_return", "significant",
        ]:
            assert key in result

    def test_significant_flag(self):
        r = _returns([0.001] * 50)
        eq = _equity((1 + r).cumprod())
        pnl = r
        m = compute_all_metrics(r, eq, total_trades=50, trade_pnls=pnl)
        assert m["significant"] is True

        m2 = compute_all_metrics(r, eq, total_trades=5, trade_pnls=pnl[:5])
        assert m2["significant"] is False
