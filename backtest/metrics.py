"""Performance metrics for backtesting."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


MAX_PLAUSIBLE_SHARPE = 6.0
MIN_TRADES_FOR_SIGNIFICANCE = 30


def compute_sharpe(returns: pd.Series, periods_per_year: int = 252) -> float:
    if returns.std() == 0:
        return 0.0
    sharpe = float(np.sqrt(periods_per_year) * returns.mean() / returns.std())
    return min(sharpe, MAX_PLAUSIBLE_SHARPE)


def compute_sortino(returns: pd.Series, periods_per_year: int = 252) -> float:
    downside = returns[returns < 0]
    if len(downside) == 0 or downside.std() == 0:
        return 0.0
    return float(np.sqrt(periods_per_year) * returns.mean() / downside.std())


def compute_max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    drawdown = (equity - peak) / peak
    return float(drawdown.min()) if len(drawdown) > 0 else 0.0


def compute_calmar(returns: pd.Series, equity: pd.Series, periods_per_year: int = 252) -> float:
    annual_return = returns.mean() * periods_per_year
    max_dd = abs(compute_max_drawdown(equity))
    if max_dd == 0:
        return 0.0
    return float(annual_return / max_dd)


def compute_win_rate(trade_pnls: pd.Series) -> float:
    if len(trade_pnls) == 0:
        return 0.0
    return float((trade_pnls > 0).sum() / len(trade_pnls))


def compute_all_metrics(
    net_returns: pd.Series,
    equity: pd.Series,
    total_trades: int,
    trade_pnls: pd.Series,
) -> Dict[str, float]:
    """Compute all performance metrics in one pass."""
    return {
        "sharpe_ratio": round(compute_sharpe(net_returns), 4),
        "sortino_ratio": round(compute_sortino(net_returns), 4),
        "max_drawdown": round(compute_max_drawdown(equity), 4),
        "calmar_ratio": round(compute_calmar(net_returns, equity), 4),
        "total_trades": total_trades,
        "win_rate": round(compute_win_rate(trade_pnls), 4),
        "avg_trade_pnl": round(float(trade_pnls.mean()), 4) if len(trade_pnls) > 0 else 0.0,
        "total_return": round(float((equity.iloc[-1] / equity.iloc[0]) - 1), 4) if len(equity) > 1 else 0.0,
        "significant": total_trades >= MIN_TRADES_FOR_SIGNIFICANCE,
    }
