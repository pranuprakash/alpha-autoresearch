"""Tests for plays/sizing.py"""

import pytest
from plays.sizing import (
    kelly_fraction,
    fixed_fraction,
    volatility_adjusted_size,
    compute_shares,
    compute_contracts,
)


class TestKellyFraction:
    def test_basic_positive(self):
        f = kelly_fraction(0.55, 1.5, 1.0)
        assert 0 < f <= 0.05

    def test_capped_at_max(self):
        f = kelly_fraction(0.90, 3.0, 1.0, max_fraction=0.05)
        assert f <= 0.05

    def test_negative_edge_returns_minimum(self):
        f = kelly_fraction(0.10, 0.5, 1.0)
        assert f >= 0.005

    def test_zero_avg_loss_returns_default(self):
        f = kelly_fraction(0.6, 1.5, 0)
        assert f == 0.02

    def test_zero_avg_win_returns_default(self):
        f = kelly_fraction(0.6, 0, 1.0)
        assert f == 0.02


class TestFixedFraction:
    def test_two_pct(self):
        assert fixed_fraction(100_000, 0.02) == 2000.0

    def test_five_pct(self):
        assert fixed_fraction(50_000, 0.05) == 2500.0

    def test_zero(self):
        assert fixed_fraction(100_000, 0) == 0.0


class TestVolatilityAdjustedSize:
    def test_scales_correctly(self):
        # target 10% vol, instrument 20% vol → size = 50% of portfolio
        size = volatility_adjusted_size(100_000, 0.10, 0.20, max_pct=0.10)
        assert size == pytest.approx(10_000, rel=0.01)  # capped at max_pct

    def test_capped_at_max(self):
        size = volatility_adjusted_size(100_000, 0.50, 0.10, max_pct=0.05)
        assert size == 5_000.0

    def test_zero_instrument_vol_returns_default(self):
        size = volatility_adjusted_size(100_000, 0.10, 0)
        assert size == 2_000.0


class TestComputeShares:
    def test_basic(self):
        assert compute_shares(10_000, 100.0) == 100

    def test_floor_at_zero(self):
        assert compute_shares(0, 100.0) == 0

    def test_zero_price(self):
        assert compute_shares(10_000, 0) == 0

    def test_round_lots(self):
        shares = compute_shares(10_000, 7.5, round_lots=True)
        assert shares % 100 == 0


class TestComputeContracts:
    def test_basic(self):
        # $2000 capital / ($10 * 100) = 2 contracts
        assert compute_contracts(2_000, 10.0) == 2

    def test_minimum_one(self):
        assert compute_contracts(100, 10.0) == 1  # $100 can't buy $1000 contract → min 1

    def test_zero_price_returns_min(self):
        assert compute_contracts(1_000, 0) == 1
