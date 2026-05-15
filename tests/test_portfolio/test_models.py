"""Tests for portfolio/models.py"""

import pytest
from portfolio.models import Position, Portfolio


def make_equity_pos(**kwargs) -> Position:
    defaults = {
        "asset_type": "equity",
        "symbol": "NVDA",
        "shares": 10.0,
        "cost_basis": 850.0,
    }
    defaults.update(kwargs)
    return Position(**defaults)


def make_option_pos(**kwargs) -> Position:
    defaults = {
        "asset_type": "option",
        "symbol": "META",
        "option_type": "call",
        "strike": 700.0,
        "expiry": "2026-10-16",
        "quantity": 1,
        "cost_basis": 31.75,
    }
    defaults.update(kwargs)
    return Position(**defaults)


class TestPosition:
    def test_from_dict_equity(self):
        d = {"asset_type": "equity", "symbol": "AAPL", "shares": 5, "cost_basis": 180.0}
        pos = Position.from_dict(d)
        assert pos.symbol == "AAPL"
        assert pos.shares == 5

    def test_from_dict_ignores_unknown_keys(self):
        d = {"asset_type": "equity", "symbol": "AAPL", "shares": 5,
             "cost_basis": 180.0, "unknown_field": "garbage"}
        pos = Position.from_dict(d)
        assert pos.symbol == "AAPL"

    def test_to_dict_omits_none(self):
        pos = make_equity_pos()
        d = pos.to_dict()
        assert "delta" not in d
        assert "dte" not in d

    def test_notional_value_equity(self):
        pos = make_equity_pos()
        pos.current_value = 9000.0
        assert pos.notional_value == 9000.0

    def test_notional_value_fallback_cost_basis(self):
        pos = make_equity_pos()
        assert pos.notional_value == 8500.0  # 10 * 850

    def test_option_ticker_call(self):
        pos = make_option_pos()
        assert pos.option_ticker == "META 2026-10-16 C$700"

    def test_option_ticker_put(self):
        pos = make_option_pos(option_type="put")
        assert pos.option_ticker == "META 2026-10-16 P$700"

    def test_option_ticker_none_for_equity(self):
        pos = make_equity_pos()
        assert pos.option_ticker is None


class TestPortfolio:
    def test_empty_portfolio(self):
        p = Portfolio(positions=[], cash=5000)
        assert p.cash == 5000
        assert len(p.positions) == 0

    def test_positions_stored(self):
        pos = make_equity_pos()
        p = Portfolio(positions=[pos], cash=1000)
        assert len(p.positions) == 1
        assert p.positions[0].symbol == "NVDA"
