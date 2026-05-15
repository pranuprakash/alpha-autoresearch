"""
strategy.py — THE FILE AGENTS EDIT.

This is the single editable file in the autoresearch loop. Agents modify
this to try different trading strategies. Everything is fair game: entry/exit
logic, indicators, parameters, position sizing.

The ONLY contract: you must define generate_signals(df) -> pd.Series
where the Series contains position values (-1, 0, or +1) aligned to df's index.

Current strategy: Simple momentum crossover (baseline)
"""

import pandas as pd


# ── Parameters ──
FAST_WINDOW = 10
SLOW_WINDOW = 50


def generate_signals(df: pd.DataFrame) -> pd.Series:
    """
    Generate trading signals from OHLCV data.

    Args:
        df: DataFrame with at least a 'Close' column (and optionally
            'Open', 'High', 'Low', 'Volume')

    Returns:
        Series of positions: +1 (long), -1 (short), 0 (flat)
        Must be same length as df.
    """
    price = df["Close"] if "Close" in df.columns else df["close"]

    fast_ma = price.rolling(FAST_WINDOW, min_periods=1).mean()
    slow_ma = price.rolling(SLOW_WINDOW, min_periods=1).mean()

    signals = pd.Series(0, index=df.index)
    signals[fast_ma > slow_ma] = 1
    signals[fast_ma < slow_ma] = -1

    return signals
