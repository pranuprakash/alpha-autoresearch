"""Momentum crossover strategy template."""

import pandas as pd

FAST_WINDOW = 10
SLOW_WINDOW = 50


def generate_signals(df: pd.DataFrame) -> pd.Series:
    price = df["Close"] if "Close" in df.columns else df["close"]
    fast_ma = price.rolling(FAST_WINDOW, min_periods=1).mean()
    slow_ma = price.rolling(SLOW_WINDOW, min_periods=1).mean()

    signals = pd.Series(0, index=df.index)
    signals[fast_ma > slow_ma] = 1
    signals[fast_ma < slow_ma] = -1
    return signals
