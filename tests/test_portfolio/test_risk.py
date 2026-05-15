"""Tests for portfolio/risk.py"""

import pytest
from portfolio.models import Position, Portfolio
from portfolio.risk import compute_position_greeks, aggregate_greeks


def make_option_pos(delta=0.40, theta=-0.02, vega=0.10, gamma=0.005, qty=1) -> Position:
    return Position(
        asset_type="option",
        symbol="NVDA",
        option_type="call",
        strike=1200.0,
        expiry="2026-06-20",
        quantity=qty,
        cost_basis=9.50,
        delta=delta,
        theta=theta,
        vega=vega,
        gamma=gamma,
    )


class TestComputePositionGreeks:
    def test_scales_by_quantity_and_multiplier(self):
        pos = make_option_pos(delta=0.40, qty=2)
        g = compute_position_greeks(pos)
        # delta * qty * 100
        assert g["delta"] == pytest.approx(0.40 * 2 * 100)

    def test_theta_scales_by_quantity_only(self):
        pos = make_option_pos(theta=-0.02, qty=3)
        g = compute_position_greeks(pos)
        assert g["theta"] == pytest.approx(-0.02 * 3)

    def test_empty_for_equity(self):
        pos = Position(asset_type="equity", symbol="NVDA", shares=10)
        assert compute_position_greeks(pos) == {}

    def test_handles_none_greeks(self):
        pos = Position(
            asset_type="option", symbol="X",
            option_type="call", quantity=1,
        )
        g = compute_position_greeks(pos)
        assert g["delta"] == 0.0


class TestAggregateGreeks:
    def test_single_option(self):
        pos = make_option_pos(delta=0.40, theta=-0.02, vega=0.10, qty=1)
        port = Portfolio(positions=[pos])
        g = aggregate_greeks(port)
        assert g["delta"] == pytest.approx(0.40 * 100)
        assert g["theta"] == pytest.approx(-0.02)

    def test_equity_not_included(self):
        opt = make_option_pos(delta=0.40, qty=1)
        eq = Position(asset_type="equity", symbol="META", shares=5)
        port = Portfolio(positions=[opt, eq])
        g = aggregate_greeks(port)
        assert g["delta"] == pytest.approx(0.40 * 100)

    def test_multiple_options_sum(self):
        p1 = make_option_pos(delta=0.40, theta=-0.02, qty=1)
        p2 = make_option_pos(delta=0.30, theta=-0.01, qty=2)
        port = Portfolio(positions=[p1, p2])
        g = aggregate_greeks(port)
        expected_delta = (0.40 * 1 + 0.30 * 2) * 100
        assert g["delta"] == pytest.approx(expected_delta)

    def test_empty_portfolio(self):
        port = Portfolio(positions=[])
        g = aggregate_greeks(port)
        assert g["delta"] == 0.0
        assert g["theta"] == 0.0
