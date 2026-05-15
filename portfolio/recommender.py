"""Action recommender — assess existing positions and filter new plays."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional

from .models import Portfolio, Position
from plays.models import PlayBook

logger = logging.getLogger("Recommender")


@dataclass
class PositionAction:
    symbol: str
    asset_type: str
    action: str           # HOLD | CLOSE | TRIM | ADD | ROLL | HEDGE
    reason: str
    urgency: str          # HIGH | MEDIUM | LOW
    details: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RecommendationReport:
    generated_at: str
    portfolio_summary: Dict[str, Any]
    position_actions: List[PositionAction]
    new_plays: List[Dict[str, Any]]
    risk_notes: List[str]
    capital_summary: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "portfolio_summary": self.portfolio_summary,
            "position_actions": [a.to_dict() for a in self.position_actions],
            "new_plays": self.new_plays,
            "risk_notes": self.risk_notes,
            "capital_summary": self.capital_summary,
        }


class ActionRecommender:
    """
    Assesses existing positions and filters new plays within portfolio constraints.

    Rules:
      Options:
        - DTE < 21 + profit > 50% → CLOSE (take profits)
        - DTE < 21 + loss > 30%   → CLOSE (salvage value)
        - DTE < 21 otherwise      → ROLL
        - Loss >= 50%              → CLOSE (stop out)
        - Gain >= 100%             → CLOSE (target hit)
      Equity:
        - Down >= 8%  → TRIM (review thesis)
        - Up >= 25%   → TRIM (partial profit)
        - Otherwise   → HOLD

    Play filtering:
      - Max total risk pct of portfolio
      - Max single ticker concentration
      - Must have enough cash
    """

    def __init__(
        self,
        max_total_risk_pct: float = 0.15,
        dte_close_threshold: int = 21,
        loss_close_threshold: float = -0.50,
        profit_take_threshold: float = 1.00,
        max_single_ticker_pct: float = 0.10,
    ):
        self.max_total_risk_pct = max_total_risk_pct
        self.dte_close_threshold = dte_close_threshold
        self.loss_close_threshold = loss_close_threshold
        self.profit_take_threshold = profit_take_threshold
        self.max_single_ticker_pct = max_single_ticker_pct

    def recommend(
        self,
        portfolio: Portfolio,
        playbook: Optional[PlayBook] = None,
    ) -> RecommendationReport:
        actions = self._assess_positions(portfolio)
        risk_notes = self._build_risk_notes(portfolio)

        new_plays: List[Dict[str, Any]] = []
        capital_available = portfolio.cash
        if playbook:
            new_plays, capital_available = self._filter_plays(portfolio, playbook)

        summary = {
            "total_value": portfolio.total_value,
            "total_pnl": portfolio.total_pnl,
            "total_pnl_pct": portfolio.total_pnl_pct,
            "cash": portfolio.cash,
            "net_delta": portfolio.net_delta,
            "net_theta": portfolio.net_theta,
            "net_vega": portfolio.net_vega,
            "var_95_1d": portfolio.var_95_1d,
            "position_count": len(portfolio.positions),
        }
        cap_summary = {
            "cash_available": portfolio.cash,
            "committed_to_new_plays": sum(p.get("capital_at_risk", 0) for p in new_plays),
            "remaining_after_plays": capital_available,
        }

        return RecommendationReport(
            generated_at=datetime.now().isoformat(),
            portfolio_summary=summary,
            position_actions=actions,
            new_plays=new_plays,
            risk_notes=risk_notes,
            capital_summary=cap_summary,
        )

    # ─────────────────────────────────────────────────────────────────────────

    def _assess_positions(self, portfolio: Portfolio) -> List[PositionAction]:
        actions = []
        for pos in portfolio.positions:
            a = self._assess_one(pos)
            if a:
                actions.append(a)
        return actions

    def _assess_one(self, pos: Position) -> Optional[PositionAction]:
        pnl_frac = (pos.pnl_pct or 0) / 100

        if pos.asset_type == "option":
            label = pos.option_ticker or pos.symbol

            if pos.dte is not None and pos.dte < self.dte_close_threshold:
                if pnl_frac > 0.50:
                    return PositionAction(label, "option", "CLOSE",
                                         f"DTE={pos.dte} with +{pnl_frac*100:.0f}% gain — take profits",
                                         "HIGH", "Theta decay accelerates <21 DTE; lock in gains")
                if pnl_frac < -0.30:
                    return PositionAction(label, "option", "CLOSE",
                                         f"DTE={pos.dte} with {pnl_frac*100:.0f}% loss — cut losses",
                                         "HIGH", "Near expiry at a loss; salvage remaining value")
                return PositionAction(label, "option", "ROLL",
                                      f"DTE={pos.dte} — roll out to next monthly",
                                      "MEDIUM", "Roll to same strike 30–45 DTE out")

            if pnl_frac <= self.loss_close_threshold:
                return PositionAction(label, "option", "CLOSE",
                                      f"Stop triggered: {pnl_frac*100:.0f}% loss",
                                      "HIGH", "Risk management stop — preserve capital")

            if pnl_frac >= self.profit_take_threshold:
                return PositionAction(label, "option", "CLOSE",
                                      f"Profit target hit: +{pnl_frac*100:.0f}%",
                                      "MEDIUM", "Close or take half; trail remainder")

            return PositionAction(label, "option", "HOLD",
                                  f"P&L: {pnl_frac*100:+.0f}%, DTE: {pos.dte}",
                                  "LOW")

        if pos.asset_type in ("equity", "bond"):
            if pnl_frac <= -0.08:
                return PositionAction(pos.symbol, pos.asset_type, "TRIM",
                                      f"Down {abs(pnl_frac)*100:.0f}% — review thesis",
                                      "MEDIUM", "Reduce if thesis has changed")
            if pnl_frac >= 0.25:
                return PositionAction(pos.symbol, pos.asset_type, "TRIM",
                                      f"Up {pnl_frac*100:.0f}% — partial profit",
                                      "LOW", "Trim 25-33%; let rest run")
            return PositionAction(pos.symbol, pos.asset_type, "HOLD",
                                  f"P&L: {pnl_frac*100:+.0f}%", "LOW")

        return None

    def _filter_plays(
        self, portfolio: Portfolio, playbook: PlayBook
    ) -> tuple[List[Dict[str, Any]], float]:
        total_risk_budget = portfolio.total_value * self.max_total_risk_pct
        used_risk = 0.0
        remaining = portfolio.cash
        filtered: List[Dict[str, Any]] = []

        for play in playbook.all_plays:
            ticker = play.ticker.upper()
            cap = play.capital_at_risk

            if cap > remaining:
                continue

            # Concentration check
            existing_exp = sum(
                (p.current_value or p.notional_value or 0)
                for p in portfolio.positions
                if p.symbol.upper() == ticker
            )
            if (existing_exp + cap) / max(portfolio.total_value, 1) > self.max_single_ticker_pct:
                continue

            if used_risk + cap > total_risk_budget:
                continue

            filtered.append(play.to_dict())
            used_risk += cap
            remaining -= cap

        return filtered, remaining

    def _build_risk_notes(self, portfolio: Portfolio) -> List[str]:
        notes = []

        if abs(portfolio.net_delta) > 500:
            side = "long" if portfolio.net_delta > 0 else "short"
            notes.append(
                f"High directional exposure: net delta {portfolio.net_delta:+.0f} ({side} biased)"
            )
        if portfolio.net_theta < -50:
            notes.append(
                f"Theta drag: options costing ~${abs(portfolio.net_theta):.0f}/day in time decay"
            )
        elif portfolio.net_theta > 10:
            notes.append(
                f"Positive theta: collecting ~${portfolio.net_theta:.0f}/day from time decay"
            )
        if portfolio.var_95_1d > portfolio.total_value * 0.03:
            notes.append(
                f"Elevated risk: 1-day 95% VaR ${portfolio.var_95_1d:,.0f} > 3% of portfolio"
            )
        if portfolio.total_value > 0 and portfolio.cash / portfolio.total_value < 0.05:
            notes.append("Low cash reserve (<5%) — limited dry powder for opportunities")

        expiring = [
            p for p in portfolio.positions
            if p.asset_type == "option" and p.dte is not None and p.dte < 21
        ]
        if expiring:
            labels = ", ".join(p.option_ticker or p.symbol for p in expiring)
            notes.append(f"{len(expiring)} option(s) expiring <21 DTE: {labels}")

        return notes
