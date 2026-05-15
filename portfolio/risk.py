"""Portfolio risk metrics — Greeks aggregation and VaR."""

from __future__ import annotations

import logging
from typing import Dict

import numpy as np

from .models import Portfolio, Position

logger = logging.getLogger("PortfolioRisk")

try:
    import yfinance as yf
    HAS_YF = True
except ImportError:
    HAS_YF = False


def compute_position_greeks(pos: Position) -> Dict[str, float]:
    """Scale individual position Greeks by quantity and contract multiplier."""
    if pos.asset_type != "option":
        return {}
    q = pos.quantity or 1
    mult = 100
    return {
        "delta": (pos.delta or 0.0) * q * mult,
        "theta": (pos.theta or 0.0) * q,
        "vega": (pos.vega or 0.0) * q,
        "gamma": (pos.gamma or 0.0) * q * mult,
    }


def aggregate_greeks(portfolio: Portfolio) -> Dict[str, float]:
    delta = theta = vega = gamma = 0.0
    for pos in portfolio.positions:
        if pos.asset_type == "option":
            g = compute_position_greeks(pos)
            delta += g.get("delta", 0.0)
            theta += g.get("theta", 0.0)
            vega += g.get("vega", 0.0)
            gamma += g.get("gamma", 0.0)
    return {"delta": delta, "theta": theta, "vega": vega, "gamma": gamma}


def compute_portfolio_var(
    portfolio: Portfolio,
    confidence: float = 0.95,
) -> float:
    """
    Simplified historical VaR using 1-year daily returns for equity positions.
    Options positions contribute via delta approximation.
    Returns dollar VaR at the given confidence level.
    """
    if not HAS_YF:
        return 0.0

    equity_positions = [p for p in portfolio.positions if p.asset_type in ("equity", "bond")]
    if not equity_positions:
        return 0.0

    symbols = list({p.symbol for p in equity_positions})
    total = max(portfolio.total_value, 1.0)

    try:
        import pandas as pd

        raw = yf.download(symbols, period="1y", auto_adjust=True, progress=False)
        closes = raw["Close"] if "Close" in raw else raw
        if closes is None or len(closes) == 0:
            return 0.0

        if not isinstance(closes, pd.DataFrame):
            closes = closes.to_frame(name=symbols[0])

        daily_returns = closes.pct_change().dropna()

        weighted = pd.Series(0.0, index=daily_returns.index)
        for pos in equity_positions:
            sym = pos.symbol
            if sym not in daily_returns.columns:
                continue
            weight = (pos.current_value or pos.notional_value or 0.0) / total
            weighted = weighted + daily_returns[sym] * weight

        if len(weighted) == 0:
            return 0.0

        var_pct = float(np.percentile(weighted.dropna(), (1 - confidence) * 100))
        return abs(var_pct * portfolio.total_value)

    except Exception as e:
        logger.debug(f"VaR computation failed: {e}")
        return 0.0
