"""
FSM State definitions, transition table, and FsmContext.

States represent stages in the alpha discovery pipeline.
Transitions define valid successors from each state.
FsmContext is the full in-memory + persistent state for one research run.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


class ResearchState(str, Enum):
    IDLE = "IDLE"
    UNIVERSE_SCAN = "UNIVERSE_SCAN"
    MACRO_REGIME = "MACRO_REGIME"
    TECHNICAL_SCAN = "TECHNICAL_SCAN"
    SENTIMENT = "SENTIMENT"
    OPTIONS_FLOW = "OPTIONS_FLOW"
    SIGNAL_SYNTHESIS = "SIGNAL_SYNTHESIS"
    STRATEGY_CODEGEN = "STRATEGY_CODEGEN"
    BACKTEST_VALIDATION = "BACKTEST_VALIDATION"
    ALPHA_REPORT = "ALPHA_REPORT"
    COMPLETE = "COMPLETE"
    ERROR = "ERROR"


# Valid transitions: each state maps to its legal successors
TRANSITIONS: Dict[ResearchState, List[ResearchState]] = {
    ResearchState.IDLE: [ResearchState.UNIVERSE_SCAN],
    ResearchState.UNIVERSE_SCAN: [ResearchState.MACRO_REGIME, ResearchState.ERROR],
    ResearchState.MACRO_REGIME: [ResearchState.TECHNICAL_SCAN, ResearchState.ERROR],
    ResearchState.TECHNICAL_SCAN: [ResearchState.SENTIMENT, ResearchState.ERROR],
    ResearchState.SENTIMENT: [ResearchState.OPTIONS_FLOW, ResearchState.ERROR],
    ResearchState.OPTIONS_FLOW: [ResearchState.SIGNAL_SYNTHESIS, ResearchState.ERROR],
    ResearchState.SIGNAL_SYNTHESIS: [
        ResearchState.STRATEGY_CODEGEN,
        ResearchState.COMPLETE,  # no signals found → exit gracefully
        ResearchState.ERROR,
    ],
    ResearchState.STRATEGY_CODEGEN: [ResearchState.BACKTEST_VALIDATION, ResearchState.ERROR],
    ResearchState.BACKTEST_VALIDATION: [
        ResearchState.ALPHA_REPORT,
        ResearchState.STRATEGY_CODEGEN,  # retry with different strategy
        ResearchState.ERROR,
    ],
    ResearchState.ALPHA_REPORT: [ResearchState.COMPLETE],
    ResearchState.COMPLETE: [],
    ResearchState.ERROR: [ResearchState.UNIVERSE_SCAN, ResearchState.COMPLETE],  # retry or give up
}

# Maximum retries before ERROR → COMPLETE (give up)
MAX_RETRIES_PER_STATE = 3
# Maximum strategy codegen cycles
MAX_CODEGEN_CYCLES = 4


@dataclass
class SignalRecord:
    """A single alpha signal with supporting evidence."""
    name: str
    ticker: str
    signal_type: str          # momentum | mean_reversion | breakout | options_flow | macro
    direction: str            # long | short | neutral
    confidence: float         # 0.0 – 1.0
    evidence: List[str]       # list of supporting facts
    suggested_entry: str      # e.g. "RSI < 30 + volume spike"
    suggested_exit: str       # e.g. "RSI > 60 or -5% stop"
    timeframe: str            # "short" | "medium" | "long"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StrategyProposal:
    """A strategy.py code proposal generated for a specific signal."""
    signal_name: str
    code: str                 # full strategy.py content
    description: str          # 1-line description
    val_sharpe: float = 0.0
    train_sharpe: float = 0.0
    max_drawdown: float = 0.0
    oos_ratio: float = 0.0
    backtest_passed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FsmContext:
    """
    Full persistent context for one market research FSM run.
    Written to disk after every state transition.
    """
    # Run metadata
    run_id: str = ""
    universe: List[str] = field(default_factory=list)
    topic: str = ""
    started_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    # State machine
    current_state: str = ResearchState.IDLE.value
    previous_state: str = ""
    error_count: int = 0
    codegen_cycles: int = 0
    error_message: str = ""

    # Research outputs (accumulated across states)
    data_summary: Dict[str, Any] = field(default_factory=dict)        # from UNIVERSE_SCAN
    macro_regime: Dict[str, Any] = field(default_factory=dict)        # from MACRO_REGIME
    technical_signals: List[Dict] = field(default_factory=list)       # from TECHNICAL_SCAN
    sentiment_data: Dict[str, Any] = field(default_factory=dict)      # from SENTIMENT
    options_data: Dict[str, Any] = field(default_factory=dict)        # from OPTIONS_FLOW
    alpha_signals: List[Dict] = field(default_factory=list)           # from SIGNAL_SYNTHESIS
    strategy_proposals: List[Dict] = field(default_factory=list)      # from STRATEGY_CODEGEN
    validated_strategies: List[Dict] = field(default_factory=list)    # from BACKTEST_VALIDATION
    alpha_report: Dict[str, Any] = field(default_factory=dict)        # from ALPHA_REPORT

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: Path) -> None:
        self.updated_at = time.time()
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str))

    @classmethod
    def load(cls, path: Path) -> FsmContext:
        if not path.exists():
            return cls()
        data = json.loads(path.read_text())
        ctx = cls()
        for k, v in data.items():
            if hasattr(ctx, k):
                setattr(ctx, k, v)
        return ctx

    def transition_to(self, new_state: ResearchState) -> None:
        """Validate and apply a state transition."""
        current = ResearchState(self.current_state)
        allowed = TRANSITIONS.get(current, [])
        if new_state not in allowed:
            raise ValueError(
                f"Invalid transition: {current.value} → {new_state.value}. "
                f"Allowed: {[s.value for s in allowed]}"
            )
        self.previous_state = self.current_state
        self.current_state = new_state.value

    def is_terminal(self) -> bool:
        return self.current_state in (
            ResearchState.COMPLETE.value, ResearchState.ERROR.value
        ) and ResearchState(self.current_state) == ResearchState.COMPLETE

    def record_error(self, message: str) -> None:
        self.error_count += 1
        self.error_message = message
        self.previous_state = self.current_state
        self.current_state = ResearchState.ERROR.value
