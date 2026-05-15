"""Options-flow momentum strategy template.

Uses volume as a proxy for flow intensity. In production, replace with
actual options flow data (unusual whales, CBOE).
"""

import pandas as pd

VOLUME_LOOKBACK = 10
VOLUME_THRESHOLD = 1.5  # multiple of average volume


def generate_signals(df: pd.DataFrame) -> pd.Series:
    price = df["Close"] if "Close" in df.columns else df["close"]
    volume = df["Volume"] if "Volume" in df.columns else df.get("volume", pd.Series(0, index=df.index))

    avg_volume = volume.rolling(VOLUME_LOOKBACK, min_periods=1).mean()
    volume_spike = volume > (avg_volume * VOLUME_THRESHOLD)

    price_up = price.pct_change().fillna(0) > 0

    signals = pd.Series(0, index=df.index)
    signals[volume_spike & price_up] = 1
    signals[volume_spike & ~price_up] = -1
    return signals
