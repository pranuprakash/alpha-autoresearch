"""Tests for ActionRecommender."""

import pytest
from portfolio.models import Position, Portfolio
from portfolio.recommender import ActionRecommender, PositionAction, RecommendationReport
from plays.models import PlayBook, OptionPlay, EquityPlay


def make_portfolio(cash=10_000, **kw) -> Portfolio:
    return Portfolio(positions=[], cash=cash, total_value=100_000 + cash, **kw)


def make_option_pos(
    symbol="NVDA",
    option_type="call",
    strike=1200.0,
    expiry="2026-06-20",
    dte=30,
    pnl_pct=0.0,
    pnl=0.0,
    current_value=950.0,
) -> Position:
    pos = Position(
        asset_type="option",
        symbol=symbol,
        option_type=option_type,
        strike=strike,
        expiry=expiry,
        quantity=1,
        cost_basis=9.50,
        delta=0.40,
        theta=-0.02,
        vega=0.10,
        gamma=0.005,
        dte=dte,
        pnl_pct=pnl_pct,
        pnl=pnl,
        current_value=current_value,
    )
    return pos


def make_equity_pos(symbol="META", pnl_pct=0.0, current_value=3100.0) -> Position:
    return Position(
        asset_type="equity",
        symbol=symbol,
        shares=5,
        cost_basis=620.0,
        current_price=620.0,
        current_value=current_value,
        pnl_pct=pnl_pct,
        pnl=0.0,
    )


def make_option_play(ticker="AAPL", priority=1, capital=1000.0) -> OptionPlay:
    return OptionPlay(
        ticker=ticker,
        play_type="long_call",
        action="BUY",
        option_type="call",
        strike=200.0,
        expiry="2026-06-20",
        quantity=1,
        entry_price=5.0,
        entry_limit=5.15,
        target_price=10.0,
        stop_price=2.50,
        capital_at_risk=capital,
        portfolio_pct=1.0,
        risk_reward=2.0,
        delta=0.40,
        theta=-0.01,
        vega=0.05,
        gamma=0.002,
        iv=25.0,
        iv_rank=40.0,
        rationale="test",
        signal_source="test",
        confidence=0.70,
        priority=priority,
    )


class TestAssessOptions:
    def test_hold_when_ok(self):
        pos = make_option_pos(dte=30, pnl_pct=10.0)
        rec = ActionRecommender()
        action = rec._assess_one(pos)
        assert action.action == "HOLD"

    def test_close_at_50_pct_loss(self):
        pos = make_option_pos(dte=30, pnl_pct=-50.0)
        rec = ActionRecommender()
        action = rec._assess_one(pos)
        assert action.action == "CLOSE"
        assert action.urgency == "HIGH"

    def test_close_at_100_pct_gain(self):
        pos = make_option_pos(dte=30, pnl_pct=100.0)
        rec = ActionRecommender()
        action = rec._assess_one(pos)
        assert action.action == "CLOSE"

    def test_close_near_expiry_with_profit(self):
        pos = make_option_pos(dte=15, pnl_pct=60.0)
        rec = ActionRecommender()
        action = rec._assess_one(pos)
        assert action.action == "CLOSE"
        assert action.urgency == "HIGH"

    def test_roll_near_expiry_neutral(self):
        pos = make_option_pos(dte=15, pnl_pct=5.0)
        rec = ActionRecommender()
        action = rec._assess_one(pos)
        assert action.action == "ROLL"

    def test_close_near_expiry_with_loss(self):
        pos = make_option_pos(dte=10, pnl_pct=-35.0)
        rec = ActionRecommender()
        action = rec._assess_one(pos)
        assert action.action == "CLOSE"


class TestAssessEquity:
    def test_hold_small_gain(self):
        pos = make_equity_pos(pnl_pct=5.0)
        rec = ActionRecommender()
        action = rec._assess_one(pos)
        assert action.action == "HOLD"

    def test_trim_large_gain(self):
        pos = make_equity_pos(pnl_pct=30.0)
        rec = ActionRecommender()
        action = rec._assess_one(pos)
        assert action.action == "TRIM"

    def test_trim_large_loss(self):
        pos = make_equity_pos(pnl_pct=-10.0)
        rec = ActionRecommender()
        action = rec._assess_one(pos)
        assert action.action == "TRIM"


class TestFilterPlays:
    def test_play_fits_in_cash(self):
        port = make_portfolio(cash=5_000)
        book = PlayBook(run_id="t", generated_at="2026-01-01", universe=["AAPL"],
                        portfolio_value=100_000)
        book.option_plays.append(make_option_play(ticker="AAPL", capital=1_000))
        rec = ActionRecommender()
        filtered, remaining = rec._filter_plays(port, book)
        assert len(filtered) == 1
        assert remaining == 4_000

    def test_play_skipped_insufficient_cash(self):
        port = make_portfolio(cash=500)
        book = PlayBook(run_id="t", generated_at="2026-01-01", universe=["AAPL"],
                        portfolio_value=100_000)
        book.option_plays.append(make_option_play(ticker="AAPL", capital=1_000))
        rec = ActionRecommender()
        filtered, _ = rec._filter_plays(port, book)
        assert len(filtered) == 0

    def test_concentration_check(self):
        # NVDA already makes up 20% of portfolio
        pos = make_option_pos(symbol="NVDA", current_value=20_000)
        port = Portfolio(positions=[pos], cash=10_000, total_value=100_000)
        book = PlayBook(run_id="t", generated_at="2026-01-01", universe=["NVDA"],
                        portfolio_value=100_000)
        book.option_plays.append(make_option_play(ticker="NVDA", capital=2_000))
        rec = ActionRecommender(max_single_ticker_pct=0.10)
        filtered, _ = rec._filter_plays(port, book)
        # Existing 20% > 10% threshold → skip
        assert len(filtered) == 0


class TestRecommend:
    def test_full_report_structure(self):
        pos = make_option_pos(dte=30, pnl_pct=20.0)
        port = Portfolio(positions=[pos], cash=10_000, total_value=100_000,
                         net_delta=40.0, net_theta=-2.0, net_vega=10.0)
        rec = ActionRecommender()
        report = rec.recommend(port)
        assert isinstance(report, RecommendationReport)
        assert "total_value" in report.portfolio_summary
        assert len(report.position_actions) == 1

    def test_to_dict(self):
        port = Portfolio(positions=[], cash=10_000, total_value=50_000)
        rec = ActionRecommender()
        report = rec.recommend(port)
        d = report.to_dict()
        assert "generated_at" in d
        assert "position_actions" in d
        assert "capital_summary" in d
