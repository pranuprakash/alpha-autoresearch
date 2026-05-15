"""PlayGenerator — converts a research brief into specific trade tickets."""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .models import EquityPlay, OptionPlay, PlayBook
from .sizing import fixed_fraction, compute_shares, compute_contracts
from .options import fetch_options_chain, select_strike_by_delta, compute_iv_rank

logger = logging.getLogger("PlayGenerator")

try:
    import yfinance as yf
    HAS_YF = True
except ImportError:
    HAS_YF = False


def _fetch_price(ticker: str) -> Optional[float]:
    if not HAS_YF:
        return None
    try:
        t = yf.Ticker(ticker)
        try:
            fi = t.fast_info
            price = float(fi.get("last_price") or fi.get("regularMarketPrice") or 0)
        except Exception:
            price = 0.0
        if not price:
            hist = t.history(period="2d")
            price = float(hist["Close"].iloc[-1]) if not hist.empty else 0.0
        return price if price > 0 else None
    except Exception as e:
        logger.warning(f"Price fetch failed for {ticker}: {e}")
        return None


class PlayGenerator:
    """
    Converts a Market Research FSM brief into specific, actionable trade tickets.

    For each alpha signal in the brief:
      - momentum/breakout/mean_reversion with direction → long call or long put
      - macro/regime signals → equity long/short
    Each play is sized to risk_pct of portfolio_value (default 2%).
    """

    def __init__(
        self,
        portfolio_value: float = 100_000.0,
        risk_pct: float = 0.02,
        max_plays: int = 10,
        min_rr: float = 1.5,
        target_delta: float = 0.40,
        dte_range: tuple = (20, 60),
        prefer_options: bool = True,
    ):
        self.portfolio_value = portfolio_value
        self.risk_pct = risk_pct
        self.max_plays = max_plays
        self.min_rr = min_rr
        self.target_delta = target_delta
        self.dte_range = dte_range
        self.prefer_options = prefer_options

    def generate(
        self,
        brief_path: Optional[Path] = None,
        brief: Optional[Dict[str, Any]] = None,
    ) -> PlayBook:
        """Generate a PlayBook from a research brief file or dict."""
        if brief is None:
            if brief_path is None:
                raise ValueError("Provide brief_path or brief dict")
            brief = json.loads(brief_path.read_text())

        signals: List[Dict[str, Any]] = brief.get("alpha_signals") or brief.get("signals") or []
        run_id = brief.get("run_id", str(uuid.uuid4())[:8])
        universe = brief.get("universe", [s.get("ticker", "") for s in signals])

        book = PlayBook(
            run_id=run_id,
            generated_at=datetime.now().isoformat(),
            universe=universe,
            portfolio_value=self.portfolio_value,
        )

        priority = 1
        for signal in sorted(signals, key=lambda s: s.get("confidence", 0), reverse=True):
            if priority > self.max_plays:
                break

            ticker = signal.get("ticker", "").upper().strip()
            if not ticker:
                continue

            direction = signal.get("direction", "long")
            signal_type = signal.get("signal_type", "momentum")
            confidence = float(signal.get("confidence", 0.5))
            evidence = signal.get("evidence", [])
            rationale = "; ".join(evidence) if evidence else signal.get("rationale", "Research signal")
            signal_source = signal.get("name", signal_type)

            price = _fetch_price(ticker)
            if price is None:
                logger.warning(f"Skipping {ticker}: price unavailable")
                book.skipped.append({"ticker": ticker, "reason": "price_fetch_failed"})
                continue

            capital = fixed_fraction(self.portfolio_value, self.risk_pct)

            use_options = self.prefer_options and signal_type not in ("macro", "regime")
            if use_options:
                play: Optional[Union[OptionPlay, EquityPlay]] = self._build_option_play(
                    ticker, direction, signal_type, confidence, rationale,
                    signal_source, price, capital, priority,
                )
            else:
                play = self._build_equity_play(
                    ticker, direction, signal_type, confidence, rationale,
                    signal_source, price, capital, priority,
                )

            if play is None:
                book.skipped.append({"ticker": ticker, "reason": "no_liquid_contract"})
                continue

            if play.risk_reward < self.min_rr:
                book.skipped.append({"ticker": ticker, "reason": f"low_rr_{play.risk_reward:.1f}"})
                continue

            if isinstance(play, OptionPlay):
                book.option_plays.append(play)
            else:
                book.equity_plays.append(play)

            priority += 1

        return book

    # ─────────────────────────────────────────────────────────────────────────

    def _build_option_play(
        self,
        ticker: str,
        direction: str,
        signal_type: str,
        confidence: float,
        rationale: str,
        signal_source: str,
        price: float,
        capital: float,
        priority: int,
    ) -> Optional[Union[OptionPlay, EquityPlay]]:
        option_type = "call" if direction in ("long", "bullish") else "put"
        chain_data = fetch_options_chain(ticker, self.dte_range)

        if chain_data is None:
            return self._build_equity_play(
                ticker, direction, signal_type, confidence, rationale,
                signal_source, price, capital, priority,
            )

        df = chain_data["calls"] if option_type == "call" else chain_data["puts"]
        expiry = chain_data["expiry"]

        contract = select_strike_by_delta(df, self.target_delta, option_type, price, expiry)
        if contract is None:
            return self._build_equity_play(
                ticker, direction, signal_type, confidence, rationale,
                signal_source, price, capital, priority,
            )

        entry = contract["mid_price"]
        if entry <= 0:
            return None

        limit = round(entry * 1.03, 2)
        target = round(entry * 2.0, 2)   # 100% gain target
        stop = round(entry * 0.50, 2)    # 50% loss stop

        rr = (target - entry) / max(entry - stop, 0.01)

        contracts = compute_contracts(capital, entry)
        actual_risk = contracts * entry * 100
        portfolio_pct = actual_risk / self.portfolio_value * 100

        iv = contract.get("iv", 0.30)
        iv_rank = compute_iv_rank(ticker, iv)

        return OptionPlay(
            ticker=ticker,
            play_type=f"long_{option_type}",
            action="BUY",
            option_type=option_type,
            strike=contract["strike"],
            expiry=expiry,
            quantity=contracts,
            entry_price=round(entry, 2),
            entry_limit=round(limit, 2),
            target_price=round(target, 2),
            stop_price=round(stop, 2),
            capital_at_risk=round(actual_risk, 2),
            portfolio_pct=round(portfolio_pct, 1),
            risk_reward=round(rr, 2),
            delta=round(contract.get("delta", 0.40), 3),
            theta=round(contract.get("theta", -0.01), 4),
            vega=round(contract.get("vega", 0.05), 4),
            gamma=round(contract.get("gamma", 0.005), 5),
            iv=round(iv * 100, 1),
            iv_rank=round(iv_rank, 1),
            rationale=rationale,
            signal_source=signal_source,
            confidence=round(confidence, 3),
            priority=priority,
        )

    def _build_equity_play(
        self,
        ticker: str,
        direction: str,
        signal_type: str,
        confidence: float,
        rationale: str,
        signal_source: str,
        price: float,
        capital: float,
        priority: int,
    ) -> Optional[EquityPlay]:
        action = "BUY" if direction in ("long", "bullish") else "SHORT"

        # Tighter stop for breakout, wider for mean-reversion
        stop_pct = 0.04 if signal_type in ("breakout", "momentum") else 0.06
        target_pct = stop_pct * 2.5

        if action == "BUY":
            stop = round(price * (1 - stop_pct), 2)
            target = round(price * (1 + target_pct), 2)
            limit = round(price * 1.005, 2)
        else:
            stop = round(price * (1 + stop_pct), 2)
            target = round(price * (1 - target_pct), 2)
            limit = round(price * 0.995, 2)

        rr = abs(target - price) / max(abs(price - stop), 0.01)
        shares = compute_shares(capital / stop_pct, price)
        if shares <= 0:
            shares = max(1, int(capital / price))

        actual_risk = shares * price * stop_pct
        portfolio_pct = actual_risk / self.portfolio_value * 100

        return EquityPlay(
            ticker=ticker,
            play_type="long_equity" if action == "BUY" else "short_equity",
            action=action,
            shares=shares,
            entry_price=price,
            entry_limit=limit,
            target_price=target,
            stop_price=stop,
            capital_at_risk=round(actual_risk, 2),
            portfolio_pct=round(portfolio_pct, 1),
            risk_reward=round(rr, 2),
            rationale=rationale,
            signal_source=signal_source,
            confidence=round(confidence, 3),
            priority=priority,
        )


def format_playbook(book: PlayBook) -> str:
    """Pretty-print a PlayBook as a terminal report."""
    lines = []
    lines.append("=" * 65)
    lines.append(f"  TRADE TICKETS  |  run_id={book.run_id}")
    lines.append(f"  Portfolio: ${book.portfolio_value:,.0f}  |  {book.generated_at[:19]}")
    lines.append("=" * 65)

    plays = book.all_plays
    if not plays:
        lines.append("\n  No plays generated.")
        if book.skipped:
            lines.append(f"\n  Skipped: {book.skipped}")
        return "\n".join(lines)

    for p in plays:
        lines.append("")
        conf = getattr(p, "confidence", 0)
        conf_label = "HIGH" if conf > 0.7 else "MEDIUM" if conf > 0.5 else "LOW"
        lines.append(f"PRIORITY {p.priority} — {conf_label} CONFIDENCE")

        if isinstance(p, OptionPlay):
            lines.append(f"  {p.action} {p.quantity}x {p.instrument}")
            lines.append(f"  Entry: ${p.entry_price:.2f}  Limit: ${p.entry_limit:.2f}")
            lines.append(f"  Target: ${p.target_price:.2f} (+{((p.target_price/p.entry_price)-1)*100:.0f}%)  "
                         f"Stop: ${p.stop_price:.2f} (-{((1-(p.stop_price/p.entry_price)))*100:.0f}%)")
            lines.append(f"  Risk: ${p.capital_at_risk:,.0f} ({p.portfolio_pct:.1f}% portfolio)  R/R: {p.risk_reward:.1f}:1")
            lines.append(f"  IV: {p.iv:.0f}%  IV Rank: {p.iv_rank:.0f}th pct  Delta: {p.delta:.2f}  Theta: {p.theta:.3f}/day")
        else:
            lines.append(f"  {p.action} {p.shares} shares {p.ticker} @ ${p.entry_price:.2f} limit ${p.entry_limit:.2f}")
            lines.append(f"  Target: ${p.target_price:.2f} (+{abs((p.target_price/p.entry_price)-1)*100:.1f}%)  "
                         f"Stop: ${p.stop_price:.2f} ({((p.stop_price/p.entry_price)-1)*100:+.1f}%)")
            lines.append(f"  Risk: ${p.capital_at_risk:,.0f} ({p.portfolio_pct:.1f}% portfolio)  R/R: {p.risk_reward:.1f}:1")

        if p.rationale:
            lines.append(f"  Rationale: {p.rationale[:120]}")

    if book.skipped:
        lines.append("")
        lines.append(f"  Skipped ({len(book.skipped)}): " +
                     ", ".join(f"{s['ticker']} [{s['reason']}]" for s in book.skipped))

    lines.append("")
    lines.append("=" * 65)
    return "\n".join(lines)
