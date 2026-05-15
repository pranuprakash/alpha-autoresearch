"""
Vectorized Backtesting Engine — hardened with realistic fee/slippage
models and statistical significance checks.

Ported and refactored from the original AlphaAutoResearchClaw.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, Optional

import numpy as np
import pandas as pd

from .fees import FeeSchedule
from .metrics import compute_all_metrics
from .slippage import SlippageModel

logger = logging.getLogger("BacktestEngine")


class VectorizedBacktester:
    """
    Runs a strategy function against market data with fees, slippage, and
    time-budget guards.

    strategy_fn: takes a DataFrame, returns a Series of positions (-1, 0, +1)
    """

    TIME_BUDGET_GUARD_PCT = 0.80

    def __init__(
        self,
        data: pd.DataFrame,
        dataset_split: str,
        fee_schedule: FeeSchedule,
        slippage_model: SlippageModel,
        initial_capital: float = 100_000.0,
        time_budget_sec: Optional[float] = None,
    ):
        self.data = data.copy()
        self.dataset_split = dataset_split
        self.fees = fee_schedule
        self.slippage = slippage_model
        self.initial_capital = initial_capital
        self.time_budget_sec = time_budget_sec
        self._start_time: Optional[float] = None

    def run(
        self,
        strategy_fn: Callable[[pd.DataFrame], pd.Series],
        run_id: str = "unknown",
        strategy_name: str = "unnamed",
    ) -> Dict[str, Any]:
        self._start_time = time.monotonic()
        df = self.data.copy()

        signals = strategy_fn(df)
        if len(signals) != len(df):
            raise ValueError(
                f"Strategy returned {len(signals)} signals for {len(df)} rows."
            )

        df["signal"] = signals.values
        df["position"] = df["signal"].shift(1).fillna(0)

        price_col = "Close" if "Close" in df.columns else "close" if "close" in df.columns else "price"
        df["_price"] = df[price_col]
        df["returns"] = df["_price"].pct_change().fillna(0)
        df["trades"] = df["position"].diff().abs().fillna(0)

        vol_col = next((c for c in ["Volume", "volume"] if c in df.columns), None)

        df["slippage_bps"] = df.apply(
            lambda row: self.slippage.estimate_bps(
                abs(row.get(vol_col, 0) * row["trades"]) if vol_col else abs(row["trades"] * 100)
            ) if row["trades"] > 0 else 0.0,
            axis=1,
        )
        df["slippage_cost"] = df["slippage_bps"] / 10_000 * df["_price"] * df["trades"]
        df["fee_cost"] = df.apply(
            lambda row: self.fees.cost(abs(row["_price"] * row["trades"]))
            if row["trades"] > 0 else 0.0,
            axis=1,
        )

        df["gross_pnl"] = df["position"] * df["returns"] * self.initial_capital
        df["net_pnl"] = df["gross_pnl"] - df["slippage_cost"] - df["fee_cost"]
        df["cumulative_pnl"] = df["net_pnl"].cumsum()
        df["equity"] = self.initial_capital + df["cumulative_pnl"]

        self._check_time_budget()

        net_returns = df["net_pnl"] / self.initial_capital
        total_trades = int(df["trades"].sum())
        trade_pnls = df.loc[df["trades"] > 0, "net_pnl"]

        metrics = compute_all_metrics(net_returns, df["equity"], total_trades, trade_pnls)
        execution_time = time.monotonic() - self._start_time

        return {
            "run_id": run_id,
            "strategy_name": strategy_name,
            "dataset_split": self.dataset_split,
            **metrics,
            "total_pnl": round(float(df["cumulative_pnl"].iloc[-1]), 2) if len(df) > 0 else 0.0,
            "data_rows": len(df),
            "execution_time_sec": round(execution_time, 2),
        }

    def _check_time_budget(self) -> None:
        if self.time_budget_sec is None or self._start_time is None:
            return
        elapsed = time.monotonic() - self._start_time
        if elapsed > self.time_budget_sec * self.TIME_BUDGET_GUARD_PCT:
            raise TimeoutError(
                f"Backtest exceeded {self.TIME_BUDGET_GUARD_PCT:.0%} of "
                f"{self.time_budget_sec}s time budget."
            )
