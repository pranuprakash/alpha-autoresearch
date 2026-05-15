"""Tests for PlayGenerator in dry-run / offline mode."""

import pytest
from unittest.mock import patch, MagicMock
from plays.generator import PlayGenerator, format_playbook
from plays.models import OptionPlay, EquityPlay, PlayBook


SAMPLE_BRIEF = {
    "run_id": "test001",
    "universe": ["NVDA", "META"],
    "alpha_signals": [
        {
            "ticker": "NVDA",
            "signal_type": "momentum",
            "direction": "long",
            "confidence": 0.80,
            "evidence": ["RSI oversold", "MACD cross"],
            "name": "nvda_momentum",
        },
        {
            "ticker": "META",
            "signal_type": "mean_reversion",
            "direction": "long",
            "confidence": 0.65,
            "evidence": ["RSI bounce"],
            "name": "meta_reversion",
        },
        {
            "ticker": "SPY",
            "signal_type": "macro",
            "direction": "long",
            "confidence": 0.55,
            "evidence": ["Bull regime"],
            "name": "spy_macro",
        },
    ],
}


def _mock_price(ticker: str):
    prices = {"NVDA": 1100.0, "META": 620.0, "SPY": 550.0}
    return prices.get(ticker)


def _mock_chain(ticker, dte_range):
    import pandas as pd
    price = _mock_price(ticker) or 100.0
    strikes = [price * m for m in (0.85, 0.90, 0.95, 1.00, 1.05, 1.10)]
    # Realistic option prices: ~2-4% of stock price for ATM with 36 DTE at 30% IV
    opt_prices = [max(price * 0.025 * (1.5 - abs(k / price - 1.0)), 0.10) for k in strikes]
    calls = pd.DataFrame({
        "strike": strikes,
        "bid": [p * 0.97 for p in opt_prices],
        "ask": [p * 1.03 for p in opt_prices],
        "lastPrice": opt_prices,
        "impliedVolatility": [0.30] * 6,
        "openInterest": [200] * 6,
        "volume": [100] * 6,
    })
    return {
        "ticker": ticker,
        "price": price,
        "expiry": "2026-06-20",
        "dte": 36,
        "calls": calls,
        "puts": calls.copy(),
    }


class TestPlayGeneratorOffline:
    @pytest.fixture
    def gen(self):
        return PlayGenerator(
            portfolio_value=100_000,
            risk_pct=0.02,
            max_plays=10,
            min_rr=1.0,
            prefer_options=True,
        )

    @patch("plays.generator._fetch_price", side_effect=_mock_price)
    @patch("plays.generator.fetch_options_chain", side_effect=_mock_chain)
    @patch("plays.generator.compute_iv_rank", return_value=45.0)
    def test_generates_playbook(self, mock_iv, mock_chain, mock_price, gen):
        book = gen.generate(brief=SAMPLE_BRIEF)
        assert isinstance(book, PlayBook)
        assert book.run_id == "test001"

    @patch("plays.generator._fetch_price", side_effect=_mock_price)
    @patch("plays.generator.fetch_options_chain", side_effect=_mock_chain)
    @patch("plays.generator.compute_iv_rank", return_value=45.0)
    def test_options_plays_have_greeks(self, mock_iv, mock_chain, mock_price, gen):
        book = gen.generate(brief=SAMPLE_BRIEF)
        for p in book.option_plays:
            assert p.delta != 0
            assert p.expiry == "2026-06-20"
            assert p.quantity >= 1

    @patch("plays.generator._fetch_price", side_effect=_mock_price)
    @patch("plays.generator.fetch_options_chain", return_value=None)   # force equity fallback
    def test_fallback_to_equity_when_no_chain(self, mock_chain, mock_price, gen):
        book = gen.generate(brief=SAMPLE_BRIEF)
        assert len(book.equity_plays) > 0

    @patch("plays.generator._fetch_price", return_value=None)
    def test_skips_when_price_unavailable(self, mock_price, gen):
        book = gen.generate(brief=SAMPLE_BRIEF)
        assert len(book.skipped) == 3

    @patch("plays.generator._fetch_price", side_effect=_mock_price)
    @patch("plays.generator.fetch_options_chain", side_effect=_mock_chain)
    @patch("plays.generator.compute_iv_rank", return_value=45.0)
    def test_macro_signal_creates_equity_play(self, mock_iv, mock_chain, mock_price, gen):
        brief = {
            "run_id": "t2",
            "universe": ["SPY"],
            "alpha_signals": [{
                "ticker": "SPY",
                "signal_type": "macro",
                "direction": "long",
                "confidence": 0.70,
                "evidence": ["Bull regime"],
                "name": "macro",
            }],
        }
        book = gen.generate(brief=brief)
        # Macro signals → equity play
        assert len(book.equity_plays) >= 1

    @patch("plays.generator._fetch_price", side_effect=_mock_price)
    @patch("plays.generator.fetch_options_chain", side_effect=_mock_chain)
    @patch("plays.generator.compute_iv_rank", return_value=45.0)
    def test_priority_ordering(self, mock_iv, mock_chain, mock_price, gen):
        book = gen.generate(brief=SAMPLE_BRIEF)
        plays = book.all_plays
        priorities = [p.priority for p in plays]
        assert priorities == sorted(priorities)

    @patch("plays.generator._fetch_price", side_effect=_mock_price)
    @patch("plays.generator.fetch_options_chain", side_effect=_mock_chain)
    @patch("plays.generator.compute_iv_rank", return_value=45.0)
    def test_capital_at_risk_bounded(self, mock_iv, mock_chain, mock_price, gen):
        book = gen.generate(brief=SAMPLE_BRIEF)
        for p in book.all_plays:
            # Each play risks exactly ~2% of 100k = $2000 max (may vary by contract)
            assert p.capital_at_risk > 0
            assert p.capital_at_risk < gen.portfolio_value * 0.10

    @patch("plays.generator._fetch_price", side_effect=_mock_price)
    @patch("plays.generator.fetch_options_chain", side_effect=_mock_chain)
    @patch("plays.generator.compute_iv_rank", return_value=45.0)
    def test_format_playbook(self, mock_iv, mock_chain, mock_price, gen):
        book = gen.generate(brief=SAMPLE_BRIEF)
        text = format_playbook(book)
        assert "TRADE TICKETS" in text
        assert "PRIORITY 1" in text

    def test_empty_brief_returns_empty_book(self, gen):
        book = gen.generate(brief={"run_id": "x", "alpha_signals": []})
        assert len(book.all_plays) == 0

    @patch("plays.generator._fetch_price", side_effect=_mock_price)
    @patch("plays.generator.fetch_options_chain", side_effect=_mock_chain)
    @patch("plays.generator.compute_iv_rank", return_value=45.0)
    def test_risk_reward_above_min_rr(self, mock_iv, mock_chain, mock_price, gen):
        book = gen.generate(brief=SAMPLE_BRIEF)
        for p in book.all_plays:
            assert p.risk_reward >= gen.min_rr
