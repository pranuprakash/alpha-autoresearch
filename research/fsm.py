"""
Market Research FSM — autonomous alpha discovery pipeline.

Runs a 12-stage finite state machine that:
1. Scans a configured universe for opportunities
2. Detects macro regime
3. Runs quantitative technical indicators
4. Collects sentiment + options flow signals
5. Synthesizes ranked alpha signals (Claude)
6. Generates strategy.py proposals (Claude)
7. Backtests each proposal
8. Writes a final alpha report + research_brief.json

Each stage is resumable. State is persisted after every transition.
"""

from __future__ import annotations

import json
import logging
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .states import (
    FsmContext,
    ResearchState,
    MAX_RETRIES_PER_STATE,
    MAX_CODEGEN_CYCLES,
    SignalRecord,
    StrategyProposal,
)
from .indicators import full_indicator_scan

logger = logging.getLogger("ResearchFSM")


class MarketResearchFSM:
    """
    12-stage autonomous market research pipeline.

    Usage:
        fsm = MarketResearchFSM(project_root=Path("."), universe=["SPY","QQQ"])
        brief = fsm.run()
    """

    def __init__(
        self,
        project_root: Path,
        universe: Optional[List[str]] = None,
        topic: str = "equity alpha discovery",
        resume_run_id: Optional[str] = None,
        web_search_fn: Optional[Callable] = None,
        dry_run: bool = False,
    ):
        self.root = project_root.resolve()
        self.topic = topic
        self.dry_run = dry_run
        self.web_search_fn = web_search_fn

        artifacts = self.root / "artifacts"
        artifacts.mkdir(parents=True, exist_ok=True)

        # Load or create context
        if resume_run_id:
            ctx_path = artifacts / f"fsm_{resume_run_id}.json"
            self.ctx = FsmContext.load(ctx_path)
            logger.info(f"Resuming run {resume_run_id} from state {self.ctx.current_state}")
        else:
            run_id = str(uuid.uuid4())[:8]
            self.ctx = FsmContext(
                run_id=run_id,
                universe=universe or self._default_universe(),
                topic=topic,
            )

        self.ctx_path = artifacts / f"fsm_{self.ctx.run_id}.json"
        self.ctx.save(self.ctx_path)

    def _default_universe(self) -> List[str]:
        """Load universe from config.yaml or fall back to default."""
        cfg_path = self.root / "config.yaml"
        if cfg_path.exists():
            try:
                import yaml
                cfg = yaml.safe_load(cfg_path.read_text())
                u = cfg.get("data", {}).get("universe", [])
                if u:
                    return u
            except Exception:
                pass
        return ["SPY", "QQQ", "NVDA", "META"]

    def _save(self) -> None:
        self.ctx.save(self.ctx_path)
        logger.debug(f"Context saved: state={self.ctx.current_state} run={self.ctx.run_id}")

    def _transition(self, new_state: ResearchState) -> None:
        self.ctx.transition_to(new_state)
        self._save()
        logger.info(f"State: {self.ctx.previous_state} → {new_state.value}")

    # ─────────────────────────────────────────
    # State handlers
    # ─────────────────────────────────────────

    def _handle_universe_scan(self) -> ResearchState:
        """Download market data and run technical indicators on each ticker."""
        logger.info(f"[UNIVERSE_SCAN] Scanning {self.ctx.universe}")

        from data.providers import download_yfinance
        from backtest.splitter import TemporalDataSplitter

        import yaml
        cfg = yaml.safe_load((self.root / "config.yaml").read_text())
        data_cfg = cfg.get("data", {})
        cache_dir = self.root / data_cfg.get("cache_dir", "data/cache")
        start = data_cfg.get("period_start", "2022-01-01")
        end = data_cfg.get("period_end", "2025-12-31")

        summary: Dict[str, Any] = {}
        for ticker in self.ctx.universe:
            try:
                df = download_yfinance(
                    symbols=[ticker],
                    start=start,
                    end=end,
                    cache_dir=cache_dir,
                )
                if df is None or len(df) < 30:
                    logger.warning(f"Insufficient data for {ticker}")
                    summary[ticker] = {"error": "insufficient_data", "rows": len(df) if df is not None else 0}
                    continue

                scan = full_indicator_scan(df, ticker=ticker)
                scan["date_range"] = {
                    "start": str(df.index[0].date()),
                    "end": str(df.index[-1].date()),
                    "rows": len(df),
                }
                summary[ticker] = scan
                logger.info(
                    f"  {ticker}: regime={scan.get('regime', {}).get('regime', '?')}, "
                    f"patterns={scan.get('patterns', [])}"
                )
            except Exception as e:
                logger.warning(f"  {ticker}: scan failed — {e}")
                summary[ticker] = {"error": str(e)}

        if not summary or all("error" in v for v in summary.values()):
            raise RuntimeError("All universe scans failed — no data available")

        self.ctx.data_summary = summary
        return ResearchState.MACRO_REGIME

    def _handle_macro_regime(self) -> ResearchState:
        """Use Claude to synthesize macro regime from indicator data."""
        logger.info("[MACRO_REGIME] Synthesizing macro picture")

        if self.dry_run:
            self.ctx.macro_regime = {
                "overall_regime": "bull",
                "regime_confidence": 0.7,
                "primary_driver": "dry_run mode",
                "sector_rotation": "tech leading",
                "key_risks": ["dry_run"],
                "strategy_implications": ["momentum"],
                "macro_summary": "Dry run — skipping live LLM call.",
            }
            return ResearchState.TECHNICAL_SCAN

        from research.agents import build_macro_agent
        agent = build_macro_agent(self.root, web_search_fn=self.web_search_fn)

        # Prepare compact summary for the agent
        regime_data = {
            ticker: {
                "regime": data.get("regime", {}),
                "rsi_14": data.get("rsi_14"),
                "momentum_20d_pct": data.get("momentum_20d_pct"),
                "historical_volatility_20d": data.get("historical_volatility_20d"),
                "patterns": data.get("patterns", []),
            }
            for ticker, data in self.ctx.data_summary.items()
            if "error" not in data
        }

        msg = (
            f"Topic: {self.ctx.topic}\n"
            f"Universe: {self.ctx.universe}\n\n"
            f"Quantitative regime data:\n{json.dumps(regime_data, indent=2)}\n\n"
            f"If you want additional context, search the web for current market conditions.\n"
            f"Synthesize into an overall macro regime JSON."
        )

        result = agent.run(msg)
        if result.error:
            raise RuntimeError(f"Macro agent failed: {result.error}")

        # Parse JSON from response
        regime_json = _extract_json(result.content)
        if regime_json:
            self.ctx.macro_regime = regime_json
        else:
            # fallback if JSON extraction fails
            self.ctx.macro_regime = {
                "overall_regime": "unknown",
                "regime_confidence": 0.3,
                "macro_summary": result.content[:500],
            }

        logger.info(f"  Macro regime: {self.ctx.macro_regime.get('overall_regime')} "
                    f"(confidence={self.ctx.macro_regime.get('regime_confidence')})")
        return ResearchState.TECHNICAL_SCAN

    def _handle_technical_scan(self) -> ResearchState:
        """Extract structured technical signals from indicator data."""
        logger.info("[TECHNICAL_SCAN] Extracting technical signals")

        technical_signals = []
        for ticker, data in self.ctx.data_summary.items():
            if "error" in data:
                continue

            patterns = data.get("patterns", [])
            for pattern in patterns:
                signal = {
                    "ticker": ticker,
                    "pattern": pattern,
                    "rsi": data.get("rsi_14", 50),
                    "momentum_20d": data.get("momentum_20d_pct", 0),
                    "volume_ratio": data.get("volume_ratio_20d", 1.0),
                    "regime": data.get("regime", {}).get("regime", "unknown"),
                    "adx": data.get("adx_14", 0),
                    "support_resistance": data.get("support_resistance", {}),
                    "bollinger": data.get("bollinger_20", {}),
                }
                technical_signals.append(signal)

        self.ctx.technical_signals = technical_signals
        logger.info(f"  Found {len(technical_signals)} technical signals across {len(self.ctx.universe)} tickers")
        return ResearchState.SENTIMENT

    def _handle_sentiment(self) -> ResearchState:
        """Collect sentiment data via web search + Claude synthesis."""
        logger.info("[SENTIMENT] Collecting sentiment signals")

        if self.dry_run:
            self.ctx.sentiment_data = {
                "fear_greed_proxy": "neutral",
                "fear_greed_confidence": 0.5,
                "vix_environment": "normal",
                "market_breadth": "neutral",
                "retail_sentiment": "neutral",
                "key_sentiment_factors": ["dry_run mode"],
                "contrarian_signals": [],
            }
            return ResearchState.OPTIONS_FLOW

        from research.agents import build_sentiment_agent
        agent = build_sentiment_agent(self.root, web_search_fn=self.web_search_fn)

        tickers_str = ", ".join(self.ctx.universe)
        msg = (
            f"Analyze current market sentiment for: {tickers_str}\n"
            f"Topic: {self.ctx.topic}\n"
            f"Current macro regime: {self.ctx.macro_regime.get('overall_regime', 'unknown')}\n\n"
            f"Search for: VIX level, put/call ratio, fear/greed index, recent news sentiment. "
            f"Output sentiment JSON."
        )

        result = agent.run(msg)
        if result.error:
            logger.warning(f"Sentiment agent failed: {result.error} — using defaults")
            self.ctx.sentiment_data = {"fear_greed_proxy": "neutral", "confidence": 0.1}
        else:
            data = _extract_json(result.content)
            self.ctx.sentiment_data = data if data else {"raw": result.content[:1000]}

        return ResearchState.OPTIONS_FLOW

    def _handle_options_flow(self) -> ResearchState:
        """Collect options flow data via web search."""
        logger.info("[OPTIONS_FLOW] Scanning options activity")

        if self.dry_run:
            self.ctx.options_data = {
                "smart_money_signal": "neutral",
                "confidence": 0.5,
                "unusual_activity": [],
            }
            return ResearchState.SIGNAL_SYNTHESIS

        from research.agents import build_options_agent
        agent = build_options_agent(self.root, web_search_fn=self.web_search_fn)

        tickers_str = ", ".join(self.ctx.universe)
        msg = (
            f"Analyze options flow for: {tickers_str}\n"
            f"Search for: unusual options activity, put/call ratios, large block trades, "
            f"notable open interest changes. Output options JSON."
        )

        result = agent.run(msg)
        if result.error:
            logger.warning(f"Options agent failed: {result.error} — using defaults")
            self.ctx.options_data = {"smart_money_signal": "neutral", "confidence": 0.1}
        else:
            data = _extract_json(result.content)
            self.ctx.options_data = data if data else {"raw": result.content[:1000]}

        return ResearchState.SIGNAL_SYNTHESIS

    def _handle_signal_synthesis(self) -> ResearchState:
        """Claude synthesizes all research into ranked alpha signals."""
        logger.info("[SIGNAL_SYNTHESIS] Synthesizing alpha signals")

        if self.dry_run:
            # Generate synthetic signals for dry run
            self.ctx.alpha_signals = self._dry_run_signals()
            if not self.ctx.alpha_signals:
                return ResearchState.COMPLETE
            return ResearchState.STRATEGY_CODEGEN

        from research.agents import build_synthesis_agent
        agent = build_synthesis_agent(self.root, web_search_fn=self.web_search_fn)

        context = {
            "topic": self.ctx.topic,
            "universe": self.ctx.universe,
            "macro_regime": self.ctx.macro_regime,
            "technical_signals": self.ctx.technical_signals[:30],  # cap for context
            "sentiment": self.ctx.sentiment_data,
            "options_flow": self.ctx.options_data,
        }

        msg = (
            f"Research context:\n{json.dumps(context, indent=2, default=str)}\n\n"
            f"Synthesize all of the above into ranked alpha signals. "
            f"Be critical — only include signals with genuine edge. "
            f"Output a JSON array of signal objects."
        )

        result = agent.run(msg)
        if result.error:
            raise RuntimeError(f"Signal synthesis failed: {result.error}")

        signals = _extract_json_array(result.content)
        if signals is None:
            signals = []

        # Filter by minimum confidence
        signals = [s for s in signals if s.get("confidence", 0) >= 0.35]
        self.ctx.alpha_signals = signals

        logger.info(f"  Synthesized {len(signals)} alpha signals")
        if not signals:
            logger.info("  No high-confidence signals found — completing without strategy generation")
            return ResearchState.COMPLETE

        return ResearchState.STRATEGY_CODEGEN

    def _handle_strategy_codegen(self) -> ResearchState:
        """Generate strategy.py proposals for the top alpha signals."""
        logger.info(f"[STRATEGY_CODEGEN] Generating strategies (cycle {self.ctx.codegen_cycles + 1})")
        self.ctx.codegen_cycles += 1

        if self.ctx.codegen_cycles > MAX_CODEGEN_CYCLES:
            logger.warning("Max codegen cycles reached — proceeding to report")
            return ResearchState.ALPHA_REPORT

        # Pick the top signal (by confidence) not yet proposed
        existing_signal_names = {p.get("signal_name") for p in self.ctx.strategy_proposals}
        candidates = [
            s for s in self.ctx.alpha_signals
            if s.get("name") not in existing_signal_names
        ]
        if not candidates:
            logger.info("All signals have proposals — proceeding to backtest")
            return ResearchState.BACKTEST_VALIDATION

        signal = max(candidates, key=lambda s: s.get("confidence", 0))
        logger.info(f"  Generating strategy for signal: {signal.get('name')} ({signal.get('ticker')})")

        if self.dry_run:
            code = self._dry_run_strategy_code(signal)
            self.ctx.strategy_proposals.append({
                "signal_name": signal.get("name", "unknown"),
                "code": code,
                "description": f"Dry run strategy for {signal.get('name')}",
                "val_sharpe": 0.0,
                "backtest_passed": False,
            })
            # Queue remaining signals if any
            if len(candidates) > 1:
                return ResearchState.STRATEGY_CODEGEN
            return ResearchState.BACKTEST_VALIDATION

        from research.agents import build_codegen_agent
        agent = build_codegen_agent(self.root)

        msg = (
            f"Write a complete strategy.py for this alpha signal:\n\n"
            f"{json.dumps(signal, indent=2)}\n\n"
            f"Macro regime: {self.ctx.macro_regime.get('overall_regime', 'unknown')}\n"
            f"Remember: output ONLY the strategy.py file content."
        )

        result = agent.run(msg)
        if result.error:
            raise RuntimeError(f"Codegen agent failed: {result.error}")

        code = _extract_strategy_code(result.content)
        if not code:
            logger.warning("Could not extract strategy code — skipping this signal")
        else:
            self.ctx.strategy_proposals.append({
                "signal_name": signal.get("name", "unknown"),
                "code": code,
                "description": signal.get("evidence", ["no description"])[0][:120],
                "val_sharpe": 0.0,
                "backtest_passed": False,
            })

        # If more signals remain, come back for another codegen cycle
        remaining = [
            s for s in self.ctx.alpha_signals
            if s.get("name") not in {p["signal_name"] for p in self.ctx.strategy_proposals}
        ]
        if remaining and self.ctx.codegen_cycles < MAX_CODEGEN_CYCLES:
            return ResearchState.STRATEGY_CODEGEN

        return ResearchState.BACKTEST_VALIDATION

    def _handle_backtest_validation(self) -> ResearchState:
        """Backtest each strategy proposal using the full harness."""
        logger.info(f"[BACKTEST_VALIDATION] Testing {len(self.ctx.strategy_proposals)} proposals")

        import importlib.util
        import sys
        import tempfile
        from backtest.engine import VectorizedBacktester
        from backtest.fees import FeeSchedule
        from backtest.slippage import SlippageModel
        from backtest.splitter import TemporalDataSplitter
        from data.providers import download_yfinance
        import yaml

        cfg = yaml.safe_load((self.root / "config.yaml").read_text())
        data_cfg = cfg.get("data", {})
        bt_cfg = cfg.get("backtest", {})

        splitter = TemporalDataSplitter(
            train_pct=bt_cfg.get("train_pct", 0.60),
            val_pct=bt_cfg.get("val_pct", 0.20),
            test_pct=bt_cfg.get("test_pct", 0.20),
            embargo_pct=bt_cfg.get("embargo_pct", 0.02),
        )
        fee_model = FeeSchedule.from_name(bt_cfg.get("fee_model", "equity"))
        slippage = SlippageModel()

        for proposal in self.ctx.strategy_proposals:
            if proposal.get("backtest_passed"):
                continue

            signal_name = proposal["signal_name"]
            code = proposal.get("code", "")
            if not code:
                logger.warning(f"  No code for {signal_name} — skipping")
                continue

            # Determine which ticker to test on
            target_ticker = "SPY"
            for sig in self.ctx.alpha_signals:
                if sig.get("name") == signal_name:
                    target_ticker = sig.get("ticker", "SPY")
                    break

            # Get data for this ticker
            ticker_data = self.ctx.data_summary.get(target_ticker, {})
            if "error" in ticker_data or not ticker_data:
                target_ticker = list(self.ctx.data_summary.keys())[0]

            try:
                df = download_yfinance(
                    symbols=[target_ticker],
                    start=data_cfg.get("period_start", "2022-01-01"),
                    end=data_cfg.get("period_end", "2025-12-31"),
                    cache_dir=self.root / data_cfg.get("cache_dir", "data/cache"),
                )
            except Exception as e:
                logger.warning(f"  Data fetch failed for {target_ticker}: {e}")
                continue

            try:
                # Dynamically load strategy
                fn = _load_strategy_fn(code)
                if fn is None:
                    logger.warning(f"  Could not load strategy for {signal_name}")
                    continue

                splits = splitter.split(df)
                train_df = splits["train"]
                val_df = splits["validation"]

                train_bt = VectorizedBacktester(
                    data=train_df,
                    dataset_split="train",
                    fee_schedule=fee_model,
                    slippage_model=slippage,
                    initial_capital=bt_cfg.get("initial_capital", 100_000),
                )
                train_result = train_bt.run(fn, run_id=signal_name, strategy_name=signal_name)

                val_bt = VectorizedBacktester(
                    data=val_df,
                    dataset_split="val",
                    fee_schedule=fee_model,
                    slippage_model=slippage,
                    initial_capital=bt_cfg.get("initial_capital", 100_000),
                )
                val_result = val_bt.run(fn, run_id=signal_name, strategy_name=signal_name)

                train_sharpe = train_result["sharpe_ratio"]
                val_sharpe = val_result["sharpe_ratio"]
                oos_ratio = val_sharpe / max(train_sharpe, 0.001) if train_sharpe > 0 else 0.0

                proposal["val_sharpe"] = val_sharpe
                proposal["train_sharpe"] = train_sharpe
                proposal["max_drawdown"] = val_result["max_drawdown"]
                proposal["oos_ratio"] = oos_ratio
                proposal["total_trades"] = val_result["total_trades"]
                proposal["ticker_tested"] = target_ticker

                risk_cfg = cfg.get("risk", {})
                passed = (
                    val_sharpe > 0.1 and
                    oos_ratio >= risk_cfg.get("min_oos_ratio", 0.50) and
                    val_result["total_trades"] >= risk_cfg.get("min_trades", 30)
                )
                proposal["backtest_passed"] = passed

                logger.info(
                    f"  {signal_name} ({target_ticker}): "
                    f"train={train_sharpe:.3f} val={val_sharpe:.3f} "
                    f"oos={oos_ratio:.2f} trades={val_result['total_trades']} "
                    f"→ {'PASS' if passed else 'FAIL'}"
                )

            except Exception as e:
                logger.warning(f"  Backtest failed for {signal_name}: {e}")
                proposal["backtest_error"] = str(e)

        self.ctx.validated_strategies = [
            p for p in self.ctx.strategy_proposals if p.get("backtest_passed")
        ]

        logger.info(f"  Validated: {len(self.ctx.validated_strategies)}/{len(self.ctx.strategy_proposals)} proposals passed")
        return ResearchState.ALPHA_REPORT

    def _handle_alpha_report(self) -> ResearchState:
        """Generate the final alpha report and research_brief.json."""
        logger.info("[ALPHA_REPORT] Generating report")

        validated = self.ctx.validated_strategies
        proposals = self.ctx.strategy_proposals

        # Sort by val_sharpe descending
        proposals_sorted = sorted(proposals, key=lambda p: p.get("val_sharpe", -999), reverse=True)

        report = {
            "run_id": self.ctx.run_id,
            "topic": self.ctx.topic,
            "universe": self.ctx.universe,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "macro_regime": self.ctx.macro_regime,
            "sentiment_summary": {
                "fear_greed": self.ctx.sentiment_data.get("fear_greed_proxy", "unknown"),
                "vix_env": self.ctx.sentiment_data.get("vix_environment", "unknown"),
            },
            "options_summary": {
                "smart_money": self.ctx.options_data.get("smart_money_signal", "neutral"),
                "confidence": self.ctx.options_data.get("confidence", 0),
            },
            "alpha_signals": self.ctx.alpha_signals,
            "strategy_results": proposals_sorted,
            "top_strategy": proposals_sorted[0] if proposals_sorted else None,
            "validated_count": len(validated),
            "total_proposals": len(proposals),
        }

        self.ctx.alpha_report = report

        # Write research_brief.json for the autoresearch loop to consume
        brief_path = self.root / "artifacts" / "research_brief.json"
        brief_path.write_text(json.dumps(report, indent=2, default=str))
        logger.info(f"  Research brief written: {brief_path}")

        # Also write the best strategy.py if we have one
        if validated:
            best = max(validated, key=lambda p: p.get("val_sharpe", 0))
            best_code = best.get("code", "")
            if best_code:
                strategy_path = self.root / "strategy.py"
                # Preserve original — write to a candidate file first
                candidate_path = self.root / "artifacts" / "strategy_candidate.py"
                candidate_path.write_text(best_code)
                logger.info(
                    f"  Best strategy ({best['signal_name']}) val_sharpe={best.get('val_sharpe', 0):.4f} "
                    f"written to artifacts/strategy_candidate.py"
                )

        return ResearchState.COMPLETE

    # ─────────────────────────────────────────
    # Main run loop
    # ─────────────────────────────────────────

    def run(self) -> Dict[str, Any]:
        """
        Execute the FSM until COMPLETE or ERROR.
        Returns the alpha report dict.
        """
        state_handlers = {
            ResearchState.UNIVERSE_SCAN: self._handle_universe_scan,
            ResearchState.MACRO_REGIME: self._handle_macro_regime,
            ResearchState.TECHNICAL_SCAN: self._handle_technical_scan,
            ResearchState.SENTIMENT: self._handle_sentiment,
            ResearchState.OPTIONS_FLOW: self._handle_options_flow,
            ResearchState.SIGNAL_SYNTHESIS: self._handle_signal_synthesis,
            ResearchState.STRATEGY_CODEGEN: self._handle_strategy_codegen,
            ResearchState.BACKTEST_VALIDATION: self._handle_backtest_validation,
            ResearchState.ALPHA_REPORT: self._handle_alpha_report,
        }

        # Start if IDLE
        if self.ctx.current_state == ResearchState.IDLE.value:
            self._transition(ResearchState.UNIVERSE_SCAN)

        while self.ctx.current_state not in (
            ResearchState.COMPLETE.value,
        ):
            current = ResearchState(self.ctx.current_state)

            if current == ResearchState.ERROR:
                if self.ctx.error_count >= MAX_RETRIES_PER_STATE:
                    logger.error(f"Max retries exceeded. Last error: {self.ctx.error_message}")
                    self._transition(ResearchState.COMPLETE)
                    break
                logger.warning(f"Recovering from error (attempt {self.ctx.error_count}/{MAX_RETRIES_PER_STATE})")
                self._transition(ResearchState.UNIVERSE_SCAN)
                continue

            handler = state_handlers.get(current)
            if handler is None:
                logger.error(f"No handler for state {current}")
                break

            logger.info(f"Executing state: {current.value}")
            try:
                next_state = handler()
                self._transition(next_state)
            except Exception as e:
                logger.error(f"State {current.value} failed: {e}")
                logger.debug(traceback.format_exc())
                self.ctx.record_error(str(e))
                self._save()

        logger.info(
            f"FSM complete. Run ID: {self.ctx.run_id} | "
            f"Signals: {len(self.ctx.alpha_signals)} | "
            f"Strategies validated: {len(self.ctx.validated_strategies)}"
        )
        return self.ctx.alpha_report

    # ─────────────────────────────────────────
    # Dry-run helpers
    # ─────────────────────────────────────────

    def _dry_run_signals(self) -> List[Dict]:
        """Generate signals from the quantitative indicators without LLM."""
        signals = []
        for ticker, data in self.ctx.data_summary.items():
            if "error" in data:
                continue
            patterns = data.get("patterns", [])
            rsi = data.get("rsi_14", 50)
            mom = data.get("momentum_20d_pct", 0)
            regime = data.get("regime", {}).get("regime", "unknown")

            if "RSI_OVERSOLD" in patterns and "UPTREND" not in patterns:
                signals.append({
                    "name": f"{ticker}_mean_reversion",
                    "ticker": ticker,
                    "signal_type": "mean_reversion",
                    "direction": "long",
                    "confidence": 0.55,
                    "evidence": [f"RSI={rsi:.1f} (oversold)", f"regime={regime}"],
                    "suggested_entry": "RSI < 32 + close near lower Bollinger band",
                    "suggested_exit": "RSI > 55 or +3% gain or -2% stop",
                    "timeframe": "short",
                })
            if mom > 8 and "UPTREND" in patterns:
                signals.append({
                    "name": f"{ticker}_momentum",
                    "ticker": ticker,
                    "signal_type": "momentum",
                    "direction": "long",
                    "confidence": 0.60,
                    "evidence": [f"20d momentum={mom:.1f}%", f"regime={regime}"],
                    "suggested_entry": "price > 20-day high on above-average volume",
                    "suggested_exit": "momentum turns negative or trailing 5% stop",
                    "timeframe": "medium",
                })

        # Always include at least one signal for dry run testing
        if not signals:
            first_ticker = next(
                (t for t, d in self.ctx.data_summary.items() if "error" not in d),
                self.ctx.universe[0]
            )
            signals.append({
                "name": f"{first_ticker}_adaptive",
                "ticker": first_ticker,
                "signal_type": "momentum",
                "direction": "long",
                "confidence": 0.45,
                "evidence": ["dry run fallback signal"],
                "suggested_entry": "price above 50-day SMA",
                "suggested_exit": "price below 50-day SMA",
                "timeframe": "medium",
            })
        return signals

    def _dry_run_strategy_code(self, signal: Dict) -> str:
        """Generate a simple strategy code for dry run testing."""
        sig_type = signal.get("signal_type", "momentum")
        if sig_type == "mean_reversion":
            return (
                "import pandas as pd\nimport numpy as np\n\n"
                "RSI_PERIOD = 14\nRSI_ENTRY = 32\nRSI_EXIT = 60\n\n"
                "def generate_signals(df: pd.DataFrame) -> pd.Series:\n"
                "    close = df['Close']\n"
                "    delta = close.diff()\n"
                "    gain = delta.clip(lower=0).ewm(span=RSI_PERIOD, adjust=False).mean()\n"
                "    loss = (-delta.clip(upper=0)).ewm(span=RSI_PERIOD, adjust=False).mean()\n"
                "    rs = gain / loss.replace(0, 1e-10)\n"
                "    rsi = 100 - (100 / (1 + rs))\n"
                "    signals = pd.Series(0, index=df.index)\n"
                "    signals[rsi.shift(1) < RSI_ENTRY] = 1\n"
                "    signals[rsi.shift(1) > RSI_EXIT] = -1\n"
                "    return signals\n"
            )
        else:
            return (
                "import pandas as pd\nimport numpy as np\n\n"
                "FAST = 10\nSLOW = 50\nVOL_MULT = 1.2\n\n"
                "def generate_signals(df: pd.DataFrame) -> pd.Series:\n"
                "    close = df['Close']\n"
                "    fast_ma = close.rolling(FAST, min_periods=1).mean()\n"
                "    slow_ma = close.rolling(SLOW, min_periods=1).mean()\n"
                "    vol_avg = df['Volume'].rolling(20, min_periods=1).mean() if 'Volume' in df.columns else pd.Series(1, index=df.index)\n"
                "    vol_confirm = (df.get('Volume', vol_avg) >= vol_avg * VOL_MULT) if 'Volume' in df.columns else pd.Series(True, index=df.index)\n"
                "    signals = pd.Series(0, index=df.index)\n"
                "    signals[(fast_ma.shift(1) > slow_ma.shift(1)) & vol_confirm.shift(1)] = 1\n"
                "    signals[fast_ma.shift(1) < slow_ma.shift(1)] = -1\n"
                "    return signals\n"
            )


# ─────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────

def _extract_json(text: str) -> Optional[Dict]:
    """Try to extract a JSON object from a text response."""
    import re
    # Try code block first
    block = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if block:
        try:
            return json.loads(block.group(1))
        except Exception:
            pass
    # Try bare JSON object
    obj = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)?\}", text, re.DOTALL)
    if obj:
        try:
            return json.loads(obj.group())
        except Exception:
            pass
    # Try full text
    try:
        return json.loads(text.strip())
    except Exception:
        return None


def _extract_json_array(text: str) -> Optional[List]:
    """Try to extract a JSON array from a text response."""
    import re
    # Code block
    block = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
    if block:
        try:
            return json.loads(block.group(1))
        except Exception:
            pass
    # Bare array
    arr = re.search(r"\[.*\]", text, re.DOTALL)
    if arr:
        try:
            return json.loads(arr.group())
        except Exception:
            pass
    try:
        return json.loads(text.strip())
    except Exception:
        return None


def _extract_strategy_code(text: str) -> Optional[str]:
    """Extract Python strategy code from a model response."""
    import re
    # Code block
    block = re.search(r"```(?:python)?\s*(.*?)\s*```", text, re.DOTALL)
    if block:
        code = block.group(1)
        if "generate_signals" in code:
            return code

    # Bare code (if response is entirely the code)
    if "def generate_signals" in text:
        return text.strip()

    return None


def _load_strategy_fn(code: str):
    """Dynamically load a generate_signals function from code string."""
    import tempfile
    import importlib.util
    import sys

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, prefix="strategy_candidate_"
    ) as f:
        f.write(code)
        tmp_path = f.name

    try:
        spec = importlib.util.spec_from_file_location("_strategy_tmp", tmp_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return getattr(mod, "generate_signals", None)
    except Exception as e:
        logger.warning(f"Failed to load strategy code: {e}")
        return None
    finally:
        import os
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
