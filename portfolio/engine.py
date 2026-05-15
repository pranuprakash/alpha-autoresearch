"""PortfolioEngine — load, enrich, and analyze a portfolio."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from .models import Portfolio, Position
from .risk import aggregate_greeks, compute_portfolio_var
from plays.options import black_scholes_greeks

logger = logging.getLogger("PortfolioEngine")

try:
    import yfinance as yf
    HAS_YF = True
except ImportError:
    HAS_YF = False


# JSON schema example for portfolio input
PORTFOLIO_SCHEMA_EXAMPLE: Dict[str, Any] = {
    "cash": 10000,
    "positions": [
        {
            "asset_type": "equity",
            "symbol": "NVDA",
            "shares": 10,
            "cost_basis": 850.00,
        },
        {
            "asset_type": "option",
            "symbol": "META",
            "option_type": "call",
            "strike": 700,
            "expiry": "2026-10-16",
            "quantity": 1,
            "cost_basis": 31.75,
        },
        {
            "asset_type": "bond",
            "symbol": "TLT",
            "shares": 50,
            "cost_basis": 88.00,
        },
    ],
}


def _fetch_price(symbol: str) -> Optional[float]:
    if not HAS_YF:
        return None
    try:
        t = yf.Ticker(symbol)
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
        logger.warning(f"Price fetch failed for {symbol}: {e}")
        return None


def _enrich_equity(pos: Position) -> Position:
    price = _fetch_price(pos.symbol)
    if price:
        pos.current_price = price
        shares = pos.shares or 0
        pos.current_value = round(shares * price, 2)
        cost = (pos.cost_basis or price) * shares
        pos.pnl = round(pos.current_value - cost, 2)
        pos.pnl_pct = round((pos.pnl / cost * 100) if cost > 0 else 0.0, 2)
    return pos


def _enrich_option(pos: Position, r: float = 0.05) -> Position:
    ul_price = _fetch_price(pos.symbol)
    if ul_price is None:
        return pos
    pos.current_price = ul_price

    if pos.expiry:
        try:
            dte = max((datetime.strptime(pos.expiry, "%Y-%m-%d") - datetime.now()).days, 0)
            pos.dte = dte
            T = dte / 365.0
        except Exception:
            T = 30 / 365.0
            pos.dte = 30
    else:
        T = 30 / 365.0
        pos.dte = 30

    # Try live IV from options chain; fall back to 30%
    iv = 0.30
    if HAS_YF and pos.expiry:
        try:
            t = yf.Ticker(pos.symbol)
            exps = t.options or []
            if pos.expiry in exps:
                chain = t.option_chain(pos.expiry)
                df = chain.calls if pos.option_type == "call" else chain.puts
                if df is not None and len(df) > 0 and pos.strike is not None:
                    closest_idx = (df["strike"] - pos.strike).abs().idxmin()
                    row = df.loc[closest_idx]
                    chain_iv = float(row.get("impliedVolatility", 0))
                    if chain_iv > 0:
                        iv = chain_iv
        except Exception:
            pass

    pos.iv = round(iv * 100, 1)
    greeks = black_scholes_greeks(
        ul_price, pos.strike or ul_price, max(T, 0.001), r, iv,
        pos.option_type or "call",
    )
    pos.delta = round(greeks["delta"], 3)
    pos.theta = round(greeks["theta"], 4)
    pos.vega = round(greeks["vega"], 4)
    pos.gamma = round(greeks["gamma"], 5)

    qty = pos.quantity or 1
    opt_price = greeks.get("bs_price", 0.0)
    pos.current_value = round(opt_price * qty * 100, 2)

    cost_total = (pos.cost_basis or opt_price) * qty * 100
    pos.pnl = round(pos.current_value - cost_total, 2)
    pos.pnl_pct = round((pos.pnl / cost_total * 100) if cost_total > 0 else 0.0, 2)

    return pos


class PortfolioEngine:
    """
    Load a portfolio from a JSON file or dict, enrich with live market data,
    and compute aggregate Greeks + risk metrics.

    Input format: see PORTFOLIO_SCHEMA_EXAMPLE
    """

    def __init__(self, portfolio_value_override: Optional[float] = None):
        self.portfolio_value_override = portfolio_value_override

    def load(
        self,
        portfolio_path: Optional[Path] = None,
        data: Optional[Dict[str, Any]] = None,
    ) -> Portfolio:
        if data is None:
            if portfolio_path is None:
                raise ValueError("Provide portfolio_path or data dict")
            data = json.loads(portfolio_path.read_text())

        positions = [Position.from_dict(p) for p in data.get("positions", [])]
        cash = float(data.get("cash", 0))
        return Portfolio(positions=positions, cash=cash)

    def enrich(self, portfolio: Portfolio) -> Portfolio:
        """Fetch live prices, compute Greeks, and update all portfolio metrics."""
        for i, pos in enumerate(portfolio.positions):
            if pos.asset_type in ("equity", "bond"):
                portfolio.positions[i] = _enrich_equity(pos)
            elif pos.asset_type == "option":
                portfolio.positions[i] = _enrich_option(pos)

        total_cost = total_value = equity_val = option_notional = 0.0

        for pos in portfolio.positions:
            val = pos.current_value or pos.notional_value or 0.0
            if pos.asset_type in ("equity", "bond"):
                cost = (pos.cost_basis or 0) * (pos.shares or 0)
                equity_val += val
            elif pos.asset_type == "option":
                cost = (pos.cost_basis or 0) * (pos.quantity or 1) * 100
                option_notional += val
            else:
                cost = 0.0
            total_cost += cost
            total_value += val

        portfolio.total_equity = round(equity_val, 2)
        portfolio.total_options_notional = round(option_notional, 2)
        portfolio.total_value = round(total_value + portfolio.cash, 2)
        portfolio.total_cost = round(total_cost, 2)
        portfolio.total_pnl = round(portfolio.total_value - portfolio.total_cost, 2)
        portfolio.total_pnl_pct = round(
            portfolio.total_pnl / portfolio.total_cost * 100 if portfolio.total_cost > 0 else 0.0, 2
        )

        if self.portfolio_value_override:
            portfolio.total_value = self.portfolio_value_override

        greeks = aggregate_greeks(portfolio)
        portfolio.net_delta = round(greeks["delta"], 2)
        portfolio.net_theta = round(greeks["theta"], 2)
        portfolio.net_vega = round(greeks["vega"], 2)
        portfolio.net_gamma = round(greeks["gamma"], 4)
        portfolio.var_95_1d = round(compute_portfolio_var(portfolio), 2)

        return portfolio
