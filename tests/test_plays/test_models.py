"""Tests for plays/models.py"""

import pytest
from plays.models import OptionPlay, EquityPlay, PlayBook


def make_option_play(priority=1, confidence=0.75) -> OptionPlay:
    return OptionPlay(
        ticker="NVDA",
        play_type="long_call",
        action="BUY",
        option_type="call",
        strike=1200.0,
        expiry="2026-06-20",
        quantity=1,
        entry_price=9.50,
        entry_limit=9.79,
        target_price=19.00,
        stop_price=4.75,
        capital_at_risk=950.0,
        portfolio_pct=1.9,
        risk_reward=2.0,
        delta=0.40,
        theta=-0.02,
        vega=0.10,
        gamma=0.005,
        iv=32.0,
        iv_rank=45.0,
        rationale="breakout + momentum",
        signal_source="momentum",
        confidence=confidence,
        priority=priority,
    )


def make_equity_play(priority=1, confidence=0.6) -> EquityPlay:
    return EquityPlay(
        ticker="META",
        play_type="long_equity",
        action="BUY",
        shares=5,
        entry_price=620.0,
        entry_limit=623.1,
        target_price=675.0,
        stop_price=595.0,
        capital_at_risk=500.0,
        portfolio_pct=1.0,
        risk_reward=2.2,
        rationale="mean reversion",
        signal_source="mean_reversion",
        confidence=confidence,
        priority=priority,
    )


class TestOptionPlay:
    def test_instrument_format_call(self):
        p = make_option_play()
        assert p.instrument == "NVDA 2026-06-20 C$1200"

    def test_instrument_format_put(self):
        p = make_option_play()
        p.option_type = "put"
        assert p.instrument == "NVDA 2026-06-20 P$1200"

    def test_to_dict_includes_instrument(self):
        d = make_option_play().to_dict()
        assert "instrument" in d
        assert d["instrument"] == "NVDA 2026-06-20 C$1200"

    def test_to_dict_all_fields(self):
        d = make_option_play().to_dict()
        for key in ["ticker", "strike", "expiry", "delta", "iv_rank", "rationale"]:
            assert key in d

    def test_risk_reward_positive(self):
        p = make_option_play()
        assert p.risk_reward > 0


class TestEquityPlay:
    def test_to_dict(self):
        d = make_equity_play().to_dict()
        assert d["ticker"] == "META"
        assert d["action"] == "BUY"
        assert d["shares"] == 5

    def test_no_instrument_field(self):
        d = make_equity_play().to_dict()
        assert "instrument" not in d


class TestPlayBook:
    def test_all_plays_sorted_by_priority(self):
        book = PlayBook(
            run_id="test",
            generated_at="2026-01-01",
            universe=["NVDA", "META"],
            portfolio_value=100_000,
        )
        book.option_plays.append(make_option_play(priority=2))
        book.equity_plays.append(make_equity_play(priority=1))
        plays = book.all_plays
        assert plays[0].priority == 1
        assert plays[1].priority == 2

    def test_to_dict_structure(self):
        book = PlayBook(
            run_id="abc123",
            generated_at="2026-01-01",
            universe=["NVDA"],
            portfolio_value=50_000,
        )
        book.option_plays.append(make_option_play())
        d = book.to_dict()
        assert d["run_id"] == "abc123"
        assert len(d["option_plays"]) == 1
        assert d["equity_plays"] == []
        assert "instrument" in d["option_plays"][0]

    def test_empty_playbook(self):
        book = PlayBook(run_id="x", generated_at="2026-01-01",
                        universe=[], portfolio_value=100_000)
        assert book.all_plays == []
        assert book.to_dict()["option_plays"] == []
