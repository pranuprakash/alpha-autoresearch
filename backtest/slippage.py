"""Market-impact slippage model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SlippageModel:
    """
    Square-root impact model: slippage_bps = base + scale * sqrt(qty / ADV)
    """
    base_bps: float = 5.0
    scale_factor: float = 50.0
    avg_daily_volume: float = 1_000_000.0

    def estimate_bps(self, order_size: float) -> float:
        participation = order_size / max(self.avg_daily_volume, 1.0)
        return self.base_bps + self.scale_factor * np.sqrt(participation)
