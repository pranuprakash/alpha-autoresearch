"""Fee schedule models for different market types."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FeeSchedule:
    maker_fee: float = 0.0004
    taker_fee: float = 0.001
    min_fee: float = 0.0
    rebate: float = 0.0

    @classmethod
    def equity(cls) -> FeeSchedule:
        return cls(maker_fee=0.0, taker_fee=0.001, min_fee=0.0, rebate=0.0)

    @classmethod
    def polymarket(cls) -> FeeSchedule:
        return cls(maker_fee=0.0, taker_fee=0.01, min_fee=0.0, rebate=0.0)

    @classmethod
    def kalshi(cls) -> FeeSchedule:
        return cls(maker_fee=0.0, taker_fee=0.07, min_fee=0.0, rebate=0.0)

    @classmethod
    def from_name(cls, name: str) -> FeeSchedule:
        factories = {
            "equity": cls.equity,
            "polymarket": cls.polymarket,
            "kalshi": cls.kalshi,
        }
        factory = factories.get(name)
        if factory is None:
            raise ValueError(f"Unknown fee model: {name}. Options: {list(factories)}")
        return factory()

    def cost(self, notional: float, is_maker: bool = False) -> float:
        if is_maker:
            return max(notional * self.maker_fee - self.rebate, self.min_fee)
        return max(notional * self.taker_fee, self.min_fee)
