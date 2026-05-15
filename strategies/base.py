"""Strategy interface definition."""

from __future__ import annotations

from typing import Protocol

import pandas as pd


class StrategyProtocol(Protocol):
    """
    A strategy must implement generate_signals.

    Takes a DataFrame with OHLCV data, returns a Series of positions:
        +1 = long, -1 = short, 0 = flat

    The Series must have the same index as the input DataFrame.
    """

    def generate_signals(self, df: pd.DataFrame) -> pd.Series: ...
