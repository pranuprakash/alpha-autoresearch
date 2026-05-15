"""
Per-state agent builders for the market research FSM.

Each builder returns a ClaudeCliAgent configured with the correct
system prompt, tools, and tool handlers for that research stage.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("ResearchAgents")


# ─────────────────────────────────────────
# System prompts per state
# ─────────────────────────────────────────

SIGNAL_SYNTHESIS_PROMPT = """You are a senior quantitative strategist. Your job is to synthesize raw market data,
technical indicators, sentiment signals, and options activity into a ranked list of tradeable alpha signals.

## Your Output

Output a JSON array of signal objects. Each signal must have:
- name: short descriptive name
- ticker: symbol (e.g. "SPY")
- signal_type: "momentum" | "mean_reversion" | "breakout" | "options_flow" | "macro"
- direction: "long" | "short" | "neutral"
- confidence: float 0.0-1.0 (be conservative — real alpha is rare)
- evidence: list of supporting facts from the data (quote numbers)
- suggested_entry: specific entry condition
- suggested_exit: specific exit condition + stop loss
- timeframe: "short" (1-5 days) | "medium" (1-4 weeks) | "long" (1-3 months)

## Signal Quality Rules
1. Only generate signals with confidence > 0.40
2. Evidence must cite actual numbers from the data, not vague statements
3. A signal where multiple indicators agree is stronger than a single indicator
4. Be skeptical of signals that rely on look-ahead or require very precise entry timing
5. Output [] (empty array) if no high-quality signals exist — it's better to wait

## Format
Output ONLY the JSON array, no preamble or explanation.
"""

STRATEGY_CODEGEN_PROMPT = """You are an expert Python quant developer. You will receive an alpha signal description
and write a complete, working strategy.py file that implements it.

## The Contract (STRICT)
strategy.py MUST define:
  def generate_signals(df: pd.DataFrame) -> pd.Series:
- Input: DataFrame with columns: Open, High, Low, Close, Volume
- Output: Series with same index as df, values in {-1, 0, +1}

## Rules
1. Only import: pandas (pd), numpy (np) — no other packages
2. NO look-ahead bias: use .shift(1) before using indicators in signals
3. Keep the strategy simple and explainable — 30-80 lines of logic max
4. Add the 3 most critical parameters as module-level constants
5. Handle edge cases: min_periods in rolling, fillna(0) on signals
6. Write ONLY the strategy.py file content — nothing else

## Available Indicators Pattern
```python
import pandas as pd
import numpy as np

FAST_WINDOW = 10   # parameter 1
SLOW_WINDOW = 50   # parameter 2
RSI_PERIOD = 14    # parameter 3

def generate_signals(df: pd.DataFrame) -> pd.Series:
    close = df["Close"]
    # compute indicators...
    # generate signals with NO look-ahead
    signals = pd.Series(0, index=df.index)
    # ... set signals ...
    return signals
```
"""

MACRO_REGIME_PROMPT = """You are a macro strategist. You will receive quantitative regime data for multiple assets
and must synthesize it into a coherent market picture.

Output a JSON object with:
- overall_regime: "bull" | "bear" | "sideways" | "volatile"
- regime_confidence: 0.0-1.0
- primary_driver: what is driving the current regime
- sector_rotation: which sectors/assets are leading/lagging
- key_risks: list of top 3 risks to current regime
- strategy_implications: list of strategy types that work in this regime
- macro_summary: 2-3 sentence summary

Output ONLY the JSON object.
"""

AUDITOR_PROMPT = """You are a quantitative risk auditor. Review this strategy for:
1. Look-ahead bias (using future data in signals)
2. Overfitting signals (e.g. if-else chains with >5 conditions)
3. Implementation bugs (wrong shift, off-by-one, NaN handling)
4. Statistical plausibility (Sharpe > 3 on training data is suspicious)

Output: APPROVED, REVIEW_REQUIRED, or REJECTED with specific reasoning.
"""


# ─────────────────────────────────────────
# Tool schemas for research agents
# ─────────────────────────────────────────

def _make_web_search_schema() -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current market information, news, and sentiment data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query (be specific: include ticker symbols, timeframes)"
                    }
                },
                "required": ["query"]
            }
        }
    }


def _make_read_file_schema() -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a local file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path relative to project root"}
                },
                "required": ["path"]
            }
        }
    }


# ─────────────────────────────────────────
# Agent builders
# ─────────────────────────────────────────

def build_synthesis_agent(
    project_root: Path,
    web_search_fn: Optional[Callable] = None,
) -> Any:
    """Agent for SIGNAL_SYNTHESIS state."""
    from core.claude_cli import ClaudeCliAgent
    from core.tools import ToolHandlers

    tools = [_make_read_file_schema()]
    tool_handlers = ToolHandlers(project_root).get_handlers()

    if web_search_fn:
        tools.append(_make_web_search_schema())
        tool_handlers["web_search"] = web_search_fn

    return ClaudeCliAgent(
        name="SignalSynthesizer",
        model="claude-cli/claude-sonnet-4-6",
        system_prompt=SIGNAL_SYNTHESIS_PROMPT,
        tools=tools,
        tool_handlers=tool_handlers,
    )


def build_codegen_agent(project_root: Path) -> Any:
    """Agent for STRATEGY_CODEGEN state."""
    from core.claude_cli import ClaudeCliAgent
    from core.tools import ToolHandlers

    handlers = ToolHandlers(project_root).get_handlers()
    from core.tools import TOOL_SCHEMAS

    return ClaudeCliAgent(
        name="StrategyCodegen",
        model="claude-cli/claude-sonnet-4-6",
        system_prompt=STRATEGY_CODEGEN_PROMPT,
        tools=TOOL_SCHEMAS,
        tool_handlers=handlers,
    )


def build_macro_agent(
    project_root: Path,
    web_search_fn: Optional[Callable] = None,
) -> Any:
    """Agent for MACRO_REGIME state."""
    from core.claude_cli import ClaudeCliAgent
    from core.tools import ToolHandlers

    tools = [_make_read_file_schema()]
    tool_handlers = ToolHandlers(project_root).get_handlers()

    if web_search_fn:
        tools.append(_make_web_search_schema())
        tool_handlers["web_search"] = web_search_fn

    return ClaudeCliAgent(
        name="MacroStrategist",
        model="claude-cli/claude-sonnet-4-6",
        system_prompt=MACRO_REGIME_PROMPT,
        tools=tools,
        tool_handlers=tool_handlers,
    )


def build_sentiment_agent(
    project_root: Path,
    web_search_fn: Optional[Callable] = None,
) -> Any:
    """Agent for SENTIMENT state — uses web search."""
    from core.claude_cli import ClaudeCliAgent
    from core.tools import ToolHandlers

    system_prompt = """You are a market sentiment analyst. Search for current sentiment data
and output a JSON object with:
- fear_greed_proxy: "extreme_fear" | "fear" | "neutral" | "greed" | "extreme_greed"
- fear_greed_confidence: 0.0-1.0
- vix_environment: "low" (<15) | "normal" (15-25) | "elevated" (25-35) | "extreme" (>35)
- market_breadth: "strong" | "neutral" | "weak" (based on advancers/decliners if available)
- retail_sentiment: "bullish" | "neutral" | "bearish" (from news/social)
- key_sentiment_factors: list of up to 5 specific observations with sources
- contrarian_signals: list of any extreme sentiment readings that suggest a reversal

Output ONLY the JSON object."""

    tools = [_make_read_file_schema()]
    tool_handlers = ToolHandlers(project_root).get_handlers()

    if web_search_fn:
        tools.append(_make_web_search_schema())
        tool_handlers["web_search"] = web_search_fn

    return ClaudeCliAgent(
        name="SentimentAnalyst",
        model="claude-cli/claude-sonnet-4-6",
        system_prompt=system_prompt,
        tools=tools,
        tool_handlers=tool_handlers,
    )


def build_options_agent(
    project_root: Path,
    web_search_fn: Optional[Callable] = None,
) -> Any:
    """Agent for OPTIONS_FLOW state."""
    from core.claude_cli import ClaudeCliAgent
    from core.tools import ToolHandlers

    system_prompt = """You are an options flow analyst. Search for unusual options activity and output a JSON object with:
- put_call_ratio: float or null if unavailable
- put_call_interpretation: "bearish" | "neutral" | "bullish"
- unusual_activity: list of notable unusual options plays (ticker, strike, expiry, side, significance)
- skew_signal: "bearish_skew" | "neutral" | "bullish_skew"
- key_levels: list of major options strikes with significant open interest
- smart_money_signal: "net_long" | "neutral" | "net_short" based on flow
- confidence: 0.0-1.0

Output ONLY the JSON object. If data is unavailable, output {"confidence": 0.1, "smart_money_signal": "neutral"}."""

    tools = [_make_read_file_schema()]
    tool_handlers = ToolHandlers(project_root).get_handlers()

    if web_search_fn:
        tools.append(_make_web_search_schema())
        tool_handlers["web_search"] = web_search_fn

    return ClaudeCliAgent(
        name="OptionsAnalyst",
        model="claude-cli/claude-sonnet-4-6",
        system_prompt=system_prompt,
        tools=tools,
        tool_handlers=tool_handlers,
    )
