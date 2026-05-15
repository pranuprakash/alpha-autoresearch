"""Tests for PortfolioEngine — offline mode with mocked data."""

import pytest
from unittest.mock import patch
from portfolio.models import Position, Portfolio
from portfolio.engine import PortfolioEngine, _enrich_equity, _enrich_option


def mock_price(symbol: str):
    prices = {"NVDA": 1100.0, "META": 620.0, "TLT": 95.0, "SPY": 550.0}
    return prices.get(symbol)


SAMPLE_PORTFOLIO_DATA = {
    "cash": 10_000,
    "positions": [
        {"asset_type": "equity", "symbol": "NVDA", "shares": 5, "cost_basis": 900.0},
        {"asset_type": "option", "symbol": "META", "option_type": "call",
         "strike": 650.0, "expiry": "2026-10-16", "quantity": 1, "cost_basis": 20.0},
        {"asset_type": "bond", "symbol": "TLT", "shares": 20, "cost_basis": 90.0},
    ],
}


class TestPortfolioEngineLoad:
    def test_load_from_dict(self):
        eng = PortfolioEngine()
        port = eng.load(data=SAMPLE_PORTFOLIO_DATA)
        assert len(port.positions) == 3
        assert port.cash == 10_000

    def test_raises_without_inputs(self):
        eng = PortfolioEngine()
        with pytest.raises(ValueError):
            eng.load()

    def test_position_types_correct(self):
        eng = PortfolioEngine()
        port = eng.load(data=SAMPLE_PORTFOLIO_DATA)
        types = {p.symbol: p.asset_type for p in port.positions}
        assert types["NVDA"] == "equity"
        assert types["META"] == "option"
        assert types["TLT"] == "bond"


class TestEnrichEquity:
    @patch("portfolio.engine._fetch_price", side_effect=mock_price)
    def test_enriches_price(self, _):
        pos = Position(asset_type="equity", symbol="NVDA", shares=5, cost_basis=900.0)
        result = _enrich_equity(pos)
        assert result.current_price == 1100.0
        assert result.current_value == pytest.approx(5500.0)
        assert result.pnl == pytest.approx(5500.0 - 4500.0)

    @patch("portfolio.engine._fetch_price", return_value=None)
    def test_no_op_when_price_unavailable(self, _):
        pos = Position(asset_type="equity", symbol="NVDA", shares=5, cost_basis=900.0)
        result = _enrich_equity(pos)
        assert result.current_price is None


class TestEnrichOption:
    @patch("portfolio.engine._fetch_price", side_effect=mock_price)
    def test_enriches_greeks(self, _):
        pos = Position(
            asset_type="option", symbol="META", option_type="call",
            strike=650.0, expiry="2026-10-16", quantity=1, cost_basis=20.0,
        )
        result = _enrich_option(pos)
        assert result.delta is not None
        assert result.theta is not None
        assert result.dte is not None
        assert result.dte >= 0

    @patch("portfolio.engine._fetch_price", return_value=None)
    def test_no_op_when_price_unavailable(self, _):
        pos = Position(
            asset_type="option", symbol="META", option_type="call",
            strike=650.0, expiry="2026-10-16", quantity=1,
        )
        result = _enrich_option(pos)
        assert result.delta is None


class TestPortfolioEngineEnrich:
    @patch("portfolio.engine._fetch_price", side_effect=mock_price)
    def test_enrich_computes_totals(self, _):
        eng = PortfolioEngine()
        port = eng.load(data=SAMPLE_PORTFOLIO_DATA)
        enriched = eng.enrich(port)
        assert enriched.total_value > 0
        assert enriched.total_equity > 0

    @patch("portfolio.engine._fetch_price", side_effect=mock_price)
    def test_cash_included_in_total(self, _):
        eng = PortfolioEngine()
        port = eng.load(data=SAMPLE_PORTFOLIO_DATA)
        enriched = eng.enrich(port)
        assert enriched.total_value >= 10_000  # at least cash

    @patch("portfolio.engine._fetch_price", side_effect=mock_price)
    def test_override_portfolio_value(self, _):
        eng = PortfolioEngine(portfolio_value_override=200_000)
        port = eng.load(data=SAMPLE_PORTFOLIO_DATA)
        enriched = eng.enrich(port)
        assert enriched.total_value == 200_000

    @patch("portfolio.engine._fetch_price", side_effect=mock_price)
    def test_greeks_computed(self, _):
        eng = PortfolioEngine()
        port = eng.load(data=SAMPLE_PORTFOLIO_DATA)
        enriched = eng.enrich(port)
        # NET delta comes from the META call
        assert enriched.net_delta != 0.0 or enriched.net_theta != 0.0
