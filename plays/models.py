"""Trade play data models."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class OptionPlay:
    ticker: str
    play_type: str          # "long_call", "long_put", etc.
    action: str             # "BUY" or "SELL"
    option_type: str        # "call" or "put"
    strike: float
    expiry: str             # "YYYY-MM-DD"
    quantity: int
    entry_price: float
    entry_limit: float
    target_price: float
    stop_price: float
    capital_at_risk: float
    portfolio_pct: float
    risk_reward: float
    delta: float
    theta: float
    vega: float
    gamma: float
    iv: float               # current IV as percentage (e.g. 32.5 for 32.5%)
    iv_rank: float          # 0–100 percentile
    rationale: str
    signal_source: str
    confidence: float
    priority: int

    @property
    def instrument(self) -> str:
        opt = "C" if self.option_type == "call" else "P"
        return f"{self.ticker} {self.expiry} {opt}${self.strike:.0f}"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["instrument"] = self.instrument
        return d


@dataclass
class EquityPlay:
    ticker: str
    play_type: str          # "long_equity" or "short_equity"
    action: str             # "BUY" or "SHORT"
    shares: int
    entry_price: float
    entry_limit: float
    target_price: float
    stop_price: float
    capital_at_risk: float
    portfolio_pct: float
    risk_reward: float
    rationale: str
    signal_source: str
    confidence: float
    priority: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PlayBook:
    run_id: str
    generated_at: str
    universe: List[str]
    portfolio_value: float
    option_plays: List[OptionPlay] = field(default_factory=list)
    equity_plays: List[EquityPlay] = field(default_factory=list)
    skipped: List[Dict[str, str]] = field(default_factory=list)

    @property
    def all_plays(self) -> List[Any]:
        plays: List[Any] = list(self.option_plays) + list(self.equity_plays)
        return sorted(plays, key=lambda p: p.priority)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "generated_at": self.generated_at,
            "universe": self.universe,
            "portfolio_value": self.portfolio_value,
            "option_plays": [p.to_dict() for p in self.option_plays],
            "equity_plays": [p.to_dict() for p in self.equity_plays],
            "skipped": self.skipped,
        }
