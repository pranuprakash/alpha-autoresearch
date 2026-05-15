"""Mean-reversion strategy template (Bollinger Band bounce)."""

import pandas as pd

LOOKBACK = 20
NUM_STD = 2.0


def generate_signals(df: pd.DataFrame) -> pd.Series:
    price = df["Close"] if "Close" in df.columns else df["close"]

    ma = price.rolling(LOOKBACK, min_periods=1).mean()
    std = price.rolling(LOOKBACK, min_periods=1).std().fillna(0)

    upper = ma + NUM_STD * std
    lower = ma - NUM_STD * std

    signals = pd.Series(0, index=df.index)
    signals[price < lower] = 1   # oversold -> long
    signals[price > upper] = -1  # overbought -> short
    return signals
