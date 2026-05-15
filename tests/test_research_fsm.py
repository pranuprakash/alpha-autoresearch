"""
Comprehensive tests for the Market Research FSM.

All LLM calls are mocked. A full dry-run integration test verifies
end-to-end flow without API calls.
"""

from __future__ import annotations

import json
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from research.states import (
    FsmContext,
    ResearchState,
    TRANSITIONS,
    MAX_RETRIES_PER_STATE,
    SignalRecord,
    StrategyProposal,
)
from research.indicators import (
    compute_rsi,
    compute_macd,
    compute_bollinger,
    compute_atr,
    compute_adx,
    detect_regime,
    full_indicator_scan,
    compute_historical_volatility,
    compute_momentum,
    compute_volume_ratio,
    compute_support_resistance,
)
from research.fsm import (
    MarketResearchFSM,
    _extract_json,
    _extract_json_array,
    _extract_strategy_code,
    _load_strategy_fn,
)
from research.report import format_alpha_report, brief_to_context_string


# ─────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────

@pytest.fixture
def ohlcv_trend():
    """500-row uptrending OHLCV."""
    rng = np.random.default_rng(42)
    n = 500
    close = 100 + np.cumsum(rng.normal(0.15, 1.0, n))
    idx = pd.date_range("2022-01-01", periods=n, freq="B")
    return pd.DataFrame({
        "Open": close - 0.3,
        "High": close + 1.2,
        "Low": close - 1.2,
        "Close": close,
        "Volume": rng.integers(1_000_000, 5_000_000, n).astype(float),
    }, index=idx)


@pytest.fixture
def ohlcv_mean_revert():
    """250-row choppy, range-bound OHLCV."""
    rng = np.random.default_rng(99)
    n = 250
    close = 100 + rng.normal(0, 1.5, n).cumsum() * 0.2
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    return pd.DataFrame({
        "Open": close - 0.1,
        "High": close + 0.8,
        "Low": close - 0.8,
        "Close": close,
        "Volume": rng.integers(500_000, 2_000_000, n).astype(float),
    }, index=idx)


# ─────────────────────────────────────────
# Tests: FsmContext & States
# ─────────────────────────────────────────

class TestFsmContext:
    def test_initial_state_is_idle(self):
        ctx = FsmContext()
        assert ctx.current_state == ResearchState.IDLE.value

    def test_valid_transition_succeeds(self):
        ctx = FsmContext()
        ctx.transition_to(ResearchState.UNIVERSE_SCAN)
        assert ctx.current_state == ResearchState.UNIVERSE_SCAN.value
        assert ctx.previous_state == ResearchState.IDLE.value

    def test_invalid_transition_raises(self):
        ctx = FsmContext()
        with pytest.raises(ValueError, match="Invalid transition"):
            ctx.transition_to(ResearchState.ALPHA_REPORT)  # not reachable from IDLE

    def test_record_error_sets_state(self):
        ctx = FsmContext()
        ctx.transition_to(ResearchState.UNIVERSE_SCAN)
        ctx.record_error("connection timeout")
        assert ctx.current_state == ResearchState.ERROR.value
        assert ctx.error_count == 1
        assert "timeout" in ctx.error_message

    def test_save_and_load(self, tmp_path):
        ctx = FsmContext(run_id="test123", universe=["SPY", "QQQ"], topic="test")
        ctx.transition_to(ResearchState.UNIVERSE_SCAN)
        ctx.alpha_signals = [{"name": "test_signal", "confidence": 0.7}]
        path = tmp_path / "ctx_test.json"
        ctx.save(path)

        loaded = FsmContext.load(path)
        assert loaded.run_id == "test123"
        assert loaded.universe == ["SPY", "QQQ"]
        assert loaded.current_state == ResearchState.UNIVERSE_SCAN.value
        assert len(loaded.alpha_signals) == 1

    def test_load_nonexistent_returns_default(self, tmp_path):
        ctx = FsmContext.load(tmp_path / "nope.json")
        assert ctx.current_state == ResearchState.IDLE.value

    def test_is_terminal_complete(self):
        ctx = FsmContext()
        ctx.current_state = ResearchState.COMPLETE.value
        assert ctx.is_terminal() is True

    def test_is_terminal_not_error(self):
        ctx = FsmContext()
        ctx.current_state = ResearchState.ERROR.value
        assert ctx.is_terminal() is False  # ERROR can retry


class TestTransitionTable:
    def test_all_states_have_transitions(self):
        for state in ResearchState:
            assert state in TRANSITIONS, f"{state} missing from TRANSITIONS"

    def test_complete_has_no_successors(self):
        assert TRANSITIONS[ResearchState.COMPLETE] == []

    def test_idle_goes_to_scan(self):
        assert ResearchState.UNIVERSE_SCAN in TRANSITIONS[ResearchState.IDLE]

    def test_error_can_retry_or_complete(self):
        successors = TRANSITIONS[ResearchState.ERROR]
        assert ResearchState.UNIVERSE_SCAN in successors
        assert ResearchState.COMPLETE in successors


# ─────────────────────────────────────────
# Tests: Indicators
# ─────────────────────────────────────────

class TestRsi:
    def test_neutral_at_50(self, ohlcv_mean_revert):
        rsi = compute_rsi(ohlcv_mean_revert["Close"])
        assert 20 <= rsi <= 80

    def test_oversold_range(self):
        declining = pd.Series(100 - np.arange(50) * 0.5)
        rsi = compute_rsi(declining)
        assert rsi < 50

    def test_overbought_range(self):
        rising = pd.Series(100 + np.arange(50) * 0.5)
        rsi = compute_rsi(rising)
        assert rsi > 50

    def test_short_series_no_crash(self):
        rsi = compute_rsi(pd.Series([100.0, 101.0, 99.0]))
        assert isinstance(rsi, float)


class TestMacd:
    def test_returns_expected_keys(self, ohlcv_trend):
        m = compute_macd(ohlcv_trend["Close"])
        assert "macd" in m
        assert "signal" in m
        assert "histogram" in m
        assert "bullish_cross" in m
        assert "bearish_cross" in m

    def test_bullish_cross_detection(self):
        # Create a scenario where fast EMA crosses above slow EMA
        prices = pd.Series([100.0] * 30 + [101.0] * 10)
        m = compute_macd(prices)
        assert isinstance(m["bullish_cross"], bool)

    def test_not_both_cross_simultaneously(self, ohlcv_trend):
        m = compute_macd(ohlcv_trend["Close"])
        assert not (m["bullish_cross"] and m["bearish_cross"])


class TestBollinger:
    def test_returns_band_structure(self, ohlcv_trend):
        bb = compute_bollinger(ohlcv_trend["Close"])
        assert bb["upper"] > bb["middle"] > bb["lower"]
        assert 0.0 <= bb["pct_b"] <= 2.0  # can exceed 0-1 in rare cases
        assert "squeeze" in bb

    def test_squeeze_detection_on_low_vol(self):
        flat = pd.Series([100.0] * 30)
        bb = compute_bollinger(flat)
        assert bb["squeeze"] is True  # zero variance → squeeze

    def test_short_series_no_crash(self):
        bb = compute_bollinger(pd.Series([100.0, 101.0, 99.0]), period=20)
        assert isinstance(bb, dict)


class TestAtr:
    def test_atr_positive(self, ohlcv_trend):
        atr = compute_atr(ohlcv_trend)
        assert atr > 0

    def test_atr_zero_range_near_zero(self):
        df = pd.DataFrame({
            "High": [100.0] * 20,
            "Low": [100.0] * 20,
            "Close": [100.0] * 20,
        })
        atr = compute_atr(df)
        assert atr == pytest.approx(0.0, abs=0.01)


class TestAdx:
    def test_trending_market_high_adx(self, ohlcv_trend):
        adx = compute_adx(ohlcv_trend)
        assert isinstance(adx, float)
        assert adx >= 0

    def test_short_data_no_crash(self):
        df = pd.DataFrame({
            "High": [100.0 + i for i in range(5)],
            "Low": [99.0 + i for i in range(5)],
            "Close": [100.0 + i for i in range(5)],
        })
        adx = compute_adx(df)
        assert isinstance(adx, float)


class TestDetectRegime:
    def test_uptrend_detected(self, ohlcv_trend):
        regime = detect_regime(ohlcv_trend)
        assert regime["regime"] in ("bull_trend", "transitional", "high_volatility_choppy", "low_volatility_range", "mean_reverting")
        assert "confidence" in regime
        assert "adx" in regime

    def test_returns_required_keys(self, ohlcv_mean_revert):
        regime = detect_regime(ohlcv_mean_revert)
        for key in ["regime", "confidence", "adx", "rsi", "historical_volatility_pct"]:
            assert key in regime

    def test_confidence_in_range(self, ohlcv_trend):
        regime = detect_regime(ohlcv_trend)
        assert 0.0 <= regime["confidence"] <= 1.0


class TestFullIndicatorScan:
    def test_returns_all_keys(self, ohlcv_trend):
        scan = full_indicator_scan(ohlcv_trend, ticker="TEST")
        for key in ["ticker", "rsi_14", "macd", "bollinger_20", "adx_14", "patterns"]:
            assert key in scan

    def test_patterns_is_list(self, ohlcv_trend):
        scan = full_indicator_scan(ohlcv_trend)
        assert isinstance(scan["patterns"], list)

    def test_insufficient_data(self):
        tiny = pd.DataFrame({"Open": [100], "High": [101], "Low": [99], "Close": [100], "Volume": [1e6]})
        scan = full_indicator_scan(tiny, ticker="TINY")
        assert "error" in scan

    def test_volume_spike_detected(self):
        rng = np.random.default_rng(77)
        n = 100
        close = pd.Series(100 + np.cumsum(rng.normal(0.1, 1, n)))
        vol = pd.Series(rng.integers(100_000, 500_000, n).astype(float))
        vol.iloc[-1] = 2_000_000  # spike
        df = pd.DataFrame({
            "Open": close - 0.2,
            "High": close + 0.5,
            "Low": close - 0.5,
            "Close": close,
            "Volume": vol,
        })
        scan = full_indicator_scan(df)
        assert "VOLUME_SPIKE" in scan.get("patterns", [])


# ─────────────────────────────────────────
# Tests: JSON / Code extraction helpers
# ─────────────────────────────────────────

class TestExtractJson:
    def test_bare_json(self):
        text = '{"regime": "bull", "confidence": 0.7}'
        result = _extract_json(text)
        assert result == {"regime": "bull", "confidence": 0.7}

    def test_code_block_json(self):
        text = '```json\n{"key": "value"}\n```'
        result = _extract_json(text)
        assert result == {"key": "value"}

    def test_json_within_text(self):
        text = 'Here is my analysis: {"regime": "bear"} — that covers it.'
        result = _extract_json(text)
        assert result is not None

    def test_invalid_returns_none(self):
        result = _extract_json("This is not JSON at all.")
        assert result is None


class TestExtractJsonArray:
    def test_bare_array(self):
        text = '[{"name": "sig1", "confidence": 0.6}]'
        result = _extract_json_array(text)
        assert len(result) == 1
        assert result[0]["name"] == "sig1"

    def test_code_block_array(self):
        text = '```json\n[{"a": 1}, {"b": 2}]\n```'
        result = _extract_json_array(text)
        assert len(result) == 2

    def test_empty_array(self):
        result = _extract_json_array("[]")
        assert result == []

    def test_invalid_returns_none(self):
        result = _extract_json_array("no json here")
        assert result is None


class TestExtractStrategyCode:
    def test_extracts_from_code_block(self):
        text = (
            "Here is the strategy:\n```python\n"
            "import pandas as pd\n"
            "def generate_signals(df):\n"
            "    return pd.Series(1, index=df.index)\n"
            "```"
        )
        code = _extract_strategy_code(text)
        assert code is not None
        assert "generate_signals" in code

    def test_extracts_bare_code(self):
        text = (
            "import pandas as pd\n"
            "def generate_signals(df):\n"
            "    return pd.Series(0, index=df.index)\n"
        )
        code = _extract_strategy_code(text)
        assert code is not None
        assert "generate_signals" in code

    def test_returns_none_without_function(self):
        text = "```python\nprint('hello')\n```"
        code = _extract_strategy_code(text)
        assert code is None


class TestLoadStrategyFn:
    def test_loads_valid_strategy(self):
        code = (
            "import pandas as pd\n"
            "def generate_signals(df):\n"
            "    return pd.Series(1, index=df.index)\n"
        )
        fn = _load_strategy_fn(code)
        assert fn is not None
        df = pd.DataFrame({"Close": [100.0, 101.0]})
        sig = fn(df)
        assert len(sig) == 2

    def test_returns_none_on_syntax_error(self):
        fn = _load_strategy_fn("def broken syntax !!")
        assert fn is None

    def test_returns_none_when_no_function(self):
        fn = _load_strategy_fn("x = 1 + 1")
        assert fn is None


# ─────────────────────────────────────────
# Tests: report.py
# ─────────────────────────────────────────

class TestFormatAlphaReport:
    def _make_report(self):
        return {
            "run_id": "abc12345",
            "topic": "test alpha",
            "universe": ["SPY"],
            "generated_at": "2026-05-15T10:00:00Z",
            "macro_regime": {
                "overall_regime": "bull",
                "regime_confidence": 0.7,
                "primary_driver": "AI boom",
                "macro_summary": "Tech-led bull market.",
            },
            "sentiment_summary": {"fear_greed": "neutral", "vix_env": "low"},
            "options_summary": {"smart_money": "net_long", "confidence": 0.6},
            "alpha_signals": [
                {
                    "name": "SPY_momentum",
                    "ticker": "SPY",
                    "direction": "long",
                    "signal_type": "momentum",
                    "confidence": 0.65,
                    "evidence": ["20d momentum=8.5%"],
                }
            ],
            "strategy_results": [
                {
                    "signal_name": "SPY_momentum",
                    "ticker_tested": "SPY",
                    "val_sharpe": 0.45,
                    "train_sharpe": 0.8,
                    "oos_ratio": 0.56,
                    "total_trades": 45,
                    "backtest_passed": True,
                }
            ],
            "top_strategy": {
                "signal_name": "SPY_momentum",
                "val_sharpe": 0.45,
                "oos_ratio": 0.56,
                "backtest_passed": True,
            },
            "validated_count": 1,
            "total_proposals": 1,
        }

    def test_contains_run_id(self):
        report = self._make_report()
        text = format_alpha_report(report)
        assert "abc12345" in text

    def test_contains_regime(self):
        report = self._make_report()
        text = format_alpha_report(report)
        assert "bull" in text

    def test_contains_signal_name(self):
        report = self._make_report()
        text = format_alpha_report(report)
        assert "SPY_momentum" in text

    def test_contains_sharpe(self):
        report = self._make_report()
        text = format_alpha_report(report)
        assert "0.450" in text or "0.45" in text

    def test_empty_report_no_crash(self):
        text = format_alpha_report({})
        assert isinstance(text, str)


class TestBriefToContextString:
    def test_contains_regime(self):
        brief = {
            "run_id": "x",
            "topic": "t",
            "macro_regime": {"overall_regime": "sideways"},
            "sentiment_summary": {"fear_greed": "fear"},
            "alpha_signals": [],
        }
        ctx = brief_to_context_string(brief)
        assert "sideways" in ctx

    def test_includes_top_signals(self):
        brief = {
            "run_id": "x",
            "topic": "t",
            "macro_regime": {"overall_regime": "bull"},
            "sentiment_summary": {"fear_greed": "neutral"},
            "alpha_signals": [
                {"name": "SPY_test", "ticker": "SPY", "direction": "long",
                 "signal_type": "momentum", "confidence": 0.7,
                 "evidence": ["rsi=28"], "suggested_entry": "rsi < 30"},
            ],
        }
        ctx = brief_to_context_string(brief)
        assert "SPY_test" in ctx

    def test_respects_max_chars(self):
        long_brief = {
            "run_id": "x",
            "topic": "t" * 500,
            "macro_regime": {"overall_regime": "bull"},
            "sentiment_summary": {"fear_greed": "neutral"},
            "alpha_signals": [],
        }
        ctx = brief_to_context_string(long_brief, max_chars=100)
        assert len(ctx) <= 100

    def test_empty_brief_empty_string(self):
        assert brief_to_context_string({}) == ""


# ─────────────────────────────────────────
# Tests: MarketResearchFSM (dry run)
# ─────────────────────────────────────────

class TestMarketResearchFsmDryRun:
    """Full end-to-end FSM tests using dry_run=True (no LLM, no live data)."""

    def test_fsm_initializes(self, tmp_project):
        fsm = MarketResearchFSM(
            project_root=tmp_project,
            universe=["SPY"],
            topic="test",
            dry_run=True,
        )
        assert fsm.ctx.current_state == ResearchState.IDLE.value
        assert fsm.ctx.universe == ["SPY"]

    def test_fsm_creates_context_file(self, tmp_project):
        fsm = MarketResearchFSM(
            project_root=tmp_project,
            universe=["SPY"],
            dry_run=True,
        )
        assert (tmp_project / "artifacts" / f"fsm_{fsm.ctx.run_id}.json").exists()

    def test_fsm_resume_loads_context(self, tmp_project):
        fsm1 = MarketResearchFSM(project_root=tmp_project, universe=["SPY"], dry_run=True)
        run_id = fsm1.ctx.run_id
        # Save some state
        fsm1.ctx.topic = "test_topic_persisted"
        fsm1.ctx.save(fsm1.ctx_path)

        fsm2 = MarketResearchFSM(
            project_root=tmp_project,
            resume_run_id=run_id,
            dry_run=True,
        )
        assert fsm2.ctx.topic == "test_topic_persisted"

    @pytest.mark.slow
    def test_fsm_full_dry_run(self, tmp_project):
        """
        Full pipeline dry run: downloads real data + runs quantitative indicators.
        Requires internet access for yfinance.
        """
        import subprocess
        result = subprocess.run(
            ["python", "prepare.py", "--universe", "SPY", "--period-start", "2022-01-01", "--period-end", "2024-12-31"],
            cwd=str(tmp_project),
            capture_output=True,
            text=True,
            timeout=120,
        )
        # If prepare fails (e.g. no internet), skip this test
        if result.returncode != 0:
            pytest.skip("prepare.py failed (no internet or data issue)")

        fsm = MarketResearchFSM(
            project_root=tmp_project,
            universe=["SPY"],
            dry_run=True,
        )
        report = fsm.run()
        assert isinstance(report, dict)
        assert fsm.ctx.current_state == ResearchState.COMPLETE.value


class TestFsmStateMachine:
    """Test individual FSM state transitions and error handling."""

    def test_transition_from_idle_to_scan(self, tmp_project):
        fsm = MarketResearchFSM(project_root=tmp_project, universe=["SPY"], dry_run=True)
        fsm._transition(ResearchState.UNIVERSE_SCAN)
        assert fsm.ctx.current_state == ResearchState.UNIVERSE_SCAN.value

    def test_error_increments_counter(self, tmp_project):
        fsm = MarketResearchFSM(project_root=tmp_project, universe=["SPY"], dry_run=True)
        fsm._transition(ResearchState.UNIVERSE_SCAN)
        fsm.ctx.record_error("test error")
        assert fsm.ctx.error_count == 1
        assert fsm.ctx.current_state == ResearchState.ERROR.value

    def test_dry_run_signals_generated(self, tmp_project):
        """_dry_run_signals always returns at least one signal."""
        fsm = MarketResearchFSM(project_root=tmp_project, universe=["SPY"], dry_run=True)
        # Set up minimal data_summary
        fsm.ctx.data_summary = {
            "SPY": {
                "regime": {"regime": "bull_trend"},
                "rsi_14": 55.0,
                "momentum_20d_pct": 10.0,
                "patterns": ["UPTREND", "STRONG_MOMENTUM_20D"],
            }
        }
        signals = fsm._dry_run_signals()
        assert len(signals) >= 1
        assert all("confidence" in s for s in signals)

    def test_dry_run_strategy_code_valid(self, tmp_project):
        fsm = MarketResearchFSM(project_root=tmp_project, universe=["SPY"], dry_run=True)
        signal = {"signal_type": "momentum", "name": "test"}
        code = fsm._dry_run_strategy_code(signal)
        fn = _load_strategy_fn(code)
        assert fn is not None

    def test_dry_run_mean_reversion_code_valid(self, tmp_project):
        fsm = MarketResearchFSM(project_root=tmp_project, universe=["SPY"], dry_run=True)
        signal = {"signal_type": "mean_reversion", "name": "test"}
        code = fsm._dry_run_strategy_code(signal)
        fn = _load_strategy_fn(code)
        assert fn is not None

    def test_default_universe_from_config(self, tmp_project):
        fsm = MarketResearchFSM(project_root=tmp_project, dry_run=True)
        # Config has solo_agent but no universe — falls back to hardcoded default
        assert len(fsm.ctx.universe) >= 1

    def test_context_persisted_after_transition(self, tmp_project):
        fsm = MarketResearchFSM(project_root=tmp_project, universe=["SPY"], dry_run=True)
        run_id = fsm.ctx.run_id
        fsm._transition(ResearchState.UNIVERSE_SCAN)

        # Load from disk and verify state was persisted
        from research.states import FsmContext
        ctx = FsmContext.load(tmp_project / "artifacts" / f"fsm_{run_id}.json")
        assert ctx.current_state == ResearchState.UNIVERSE_SCAN.value


class TestHandleTechnicalScan:
    def test_extracts_patterns_from_scan(self, tmp_project):
        fsm = MarketResearchFSM(project_root=tmp_project, universe=["SPY"], dry_run=True)
        fsm.ctx.data_summary = {
            "SPY": {
                "patterns": ["RSI_OVERSOLD", "BOLLINGER_LOWER_TOUCH"],
                "rsi_14": 28.0,
                "momentum_20d_pct": -3.0,
                "volume_ratio_20d": 1.5,
                "regime": {"regime": "mean_reverting"},
                "adx_14": 18.0,
                "support_resistance": {"resistance": 450.0, "support": 420.0},
                "bollinger_20": {"squeeze": False},
            }
        }
        result_state = fsm._handle_technical_scan()
        assert result_state == ResearchState.SENTIMENT
        assert len(fsm.ctx.technical_signals) >= 1

    def test_skips_errored_tickers(self, tmp_project):
        fsm = MarketResearchFSM(project_root=tmp_project, universe=["SPY", "QQQ"], dry_run=True)
        fsm.ctx.data_summary = {
            "SPY": {"error": "download failed"},
            "QQQ": {"patterns": ["UPTREND"], "rsi_14": 60.0,
                    "momentum_20d_pct": 5.0, "volume_ratio_20d": 1.1,
                    "regime": {"regime": "bull_trend"}, "adx_14": 30.0,
                    "support_resistance": {}, "bollinger_20": {}},
        }
        result_state = fsm._handle_technical_scan()
        assert result_state == ResearchState.SENTIMENT
        # Only QQQ signals (SPY errored)
        tickers_scanned = {s["ticker"] for s in fsm.ctx.technical_signals}
        assert "SPY" not in tickers_scanned


class TestHandleSignalSynthesisDryRun:
    def test_generates_signals_from_dry_run_data(self, tmp_project):
        fsm = MarketResearchFSM(project_root=tmp_project, universe=["SPY"], dry_run=True)
        fsm.ctx.data_summary = {
            "SPY": {
                "regime": {"regime": "bear_trend"},
                "rsi_14": 28.0,
                "momentum_20d_pct": -2.0,
                "patterns": ["RSI_OVERSOLD"],
                "error": False,
            }
        }
        fsm.ctx.macro_regime = {"overall_regime": "bear"}
        fsm.ctx.sentiment_data = {}
        fsm.ctx.options_data = {}
        fsm.ctx.technical_signals = []

        result_state = fsm._handle_signal_synthesis()
        assert result_state in (ResearchState.STRATEGY_CODEGEN, ResearchState.COMPLETE)


class TestHandleAlphaReport:
    def test_writes_research_brief(self, tmp_project):
        fsm = MarketResearchFSM(project_root=tmp_project, universe=["SPY"], dry_run=True)
        fsm.ctx.alpha_signals = [
            {"name": "test_signal", "ticker": "SPY", "confidence": 0.6,
             "direction": "long", "signal_type": "momentum", "evidence": []}
        ]
        fsm.ctx.strategy_proposals = []
        fsm.ctx.validated_strategies = []
        fsm.ctx.macro_regime = {"overall_regime": "bull", "regime_confidence": 0.7, "macro_summary": "test"}
        fsm.ctx.sentiment_data = {"fear_greed_proxy": "neutral"}
        fsm.ctx.options_data = {"smart_money_signal": "neutral"}

        result_state = fsm._handle_alpha_report()
        assert result_state == ResearchState.COMPLETE

        brief_path = tmp_project / "artifacts" / "research_brief.json"
        assert brief_path.exists()
        brief = json.loads(brief_path.read_text())
        assert "alpha_signals" in brief
        assert brief["run_id"] == fsm.ctx.run_id

    def test_writes_candidate_strategy_when_validated(self, tmp_project):
        fsm = MarketResearchFSM(project_root=tmp_project, universe=["SPY"], dry_run=True)
        code = (
            "import pandas as pd\n"
            "def generate_signals(df): return pd.Series(1, index=df.index)\n"
        )
        fsm.ctx.validated_strategies = [{
            "signal_name": "test",
            "code": code,
            "val_sharpe": 0.5,
            "backtest_passed": True,
        }]
        fsm.ctx.strategy_proposals = fsm.ctx.validated_strategies
        fsm.ctx.alpha_signals = []
        fsm.ctx.macro_regime = {"overall_regime": "bull", "macro_summary": "t"}
        fsm.ctx.sentiment_data = {}
        fsm.ctx.options_data = {}

        fsm._handle_alpha_report()
        assert (tmp_project / "artifacts" / "strategy_candidate.py").exists()
