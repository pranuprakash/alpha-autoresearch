"""Portfolio and Position data models."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class Position:
    asset_type: str           # "equity", "option", "bond", "cash"
    symbol: str
    # Equity / bond
    shares: Optional[float] = None
    cost_basis: Optional[float] = None
    # Option-specific
    option_type: Optional[str] = None   # "call" or "put"
    strike: Optional[float] = None
    expiry: Optional[str] = None
    quantity: Optional[int] = None
    # Computed at enrichment
    current_price: Optional[float] = None
    current_value: Optional[float] = None
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    # Option Greeks
    delta: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None
    gamma: Optional[float] = None
    iv: Optional[float] = None          # IV as percentage
    dte: Optional[int] = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Position":
        valid = {k for k in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in valid})

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}

    @property
    def notional_value(self) -> float:
        if self.current_value is not None:
            return self.current_value
        if self.asset_type == "option" and self.current_price and self.quantity:
            return self.current_price * self.quantity * 100
        if self.shares and self.cost_basis:
            return self.shares * self.cost_basis
        return 0.0

    @property
    def option_ticker(self) -> Optional[str]:
        if self.asset_type != "option":
            return None
        opt = "C" if self.option_type == "call" else "P"
        return f"{self.symbol} {self.expiry} {opt}${self.strike:.0f}"


@dataclass
class Portfolio:
    positions: List[Position]
    cash: float = 0.0
    # Computed
    total_equity: float = 0.0
    total_options_notional: float = 0.0
    total_value: float = 0.0
    total_cost: float = 0.0
    total_pnl: float = 0.0
    total_pnl_pct: float = 0.0
    # Aggregate Greeks (options portfolio)
    net_delta: float = 0.0
    net_theta: float = 0.0
    net_vega: float = 0.0
    net_gamma: float = 0.0
    # Risk
    var_95_1d: float = 0.0
