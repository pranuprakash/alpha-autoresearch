#!/usr/bin/env python3
"""
prepare.py — Data prep, splits, and evaluation harness.

This is Karpathy's prepare.py equivalent. The agent NEVER modifies this file.
It downloads data, splits it, and evaluates strategy.py.

Usage:
    python prepare.py                              # Download data + show splits
    python prepare.py --evaluate                   # Run strategy.py on train+val
    python prepare.py --evaluate --split test      # One-shot test (SACRED)
    python prepare.py --universe "SPY,QQQ,AAPL"   # Custom universe
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict

import pandas as pd
import yaml

from backtest import (
    FeeSchedule,
    SlippageModel,
    TemporalDataSplitter,
    VectorizedBacktester,
)
from data.providers import load_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("prepare")

PROJECT_ROOT = Path(__file__).parent


def load_config() -> Dict[str, Any]:
    config_path = PROJECT_ROOT / "config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f)
    return {}


def load_strategy() -> Callable[[pd.DataFrame], pd.Series]:
    """Dynamically import strategy.py and return its generate_signals function."""
    strategy_path = PROJECT_ROOT / "strategy.py"
    if not strategy_path.exists():
        raise FileNotFoundError("strategy.py not found. Cannot evaluate.")

    spec = importlib.util.spec_from_file_location("strategy", str(strategy_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "generate_signals"):
        raise AttributeError(
            "strategy.py must define a 'generate_signals(df: pd.DataFrame) -> pd.Series' function."
        )
    return module.generate_signals


def prepare_data(
    universe: list[str] | None = None,
    period_start: str | None = None,
    period_end: str | None = None,
) -> pd.DataFrame:
    """Download and cache market data."""
    config = load_config()
    data_cfg = config.get("data", {})

    symbols = universe or data_cfg.get("universe", ["SPY"])
    start = period_start or data_cfg.get("period_start", "2020-01-01")
    end = period_end or data_cfg.get("period_end", "2025-12-31")
    source = data_cfg.get("source", "yfinance")
    cache_dir = Path(data_cfg.get("cache_dir", "data/cache"))

    df = load_data(symbols, start, end, source=source, cache_dir=cache_dir)
    logger.info(f"Data ready: {len(df)} rows, {df.columns.tolist()}")
    return df


def split_data(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Split data into train/val/test with embargo."""
    config = load_config()
    bt_cfg = config.get("backtest", {})

    splitter = TemporalDataSplitter(
        train_pct=bt_cfg.get("train_pct", 0.60),
        val_pct=bt_cfg.get("val_pct", 0.20),
        test_pct=bt_cfg.get("test_pct", 0.20),
        embargo_pct=bt_cfg.get("embargo_pct", 0.02),
    )
    return splitter.split(df)


def evaluate(split_name: str = "both") -> Dict[str, Any]:
    """
    Run strategy.py through the backtest engine and print results.

    split_name:
        "both"  — train + validation (default, used in the autoresearch loop)
        "train" — train only
        "val"   — validation only
        "test"  — one-shot test (SACRED)
    """
    config = load_config()
    bt_cfg = config.get("backtest", {})

    strategy_fn = load_strategy()
    df = prepare_data()

    has_symbol = "Symbol" in df.columns
    if has_symbol:
        primary_symbol = df["Symbol"].value_counts().index[0]
        df = df[df["Symbol"] == primary_symbol].copy()
        logger.info(f"Using primary symbol: {primary_symbol} ({len(df)} rows)")

    df = df.sort_index()
    splits = split_data(df)

    fee_schedule = FeeSchedule.from_name(bt_cfg.get("fee_model", "equity"))
    slippage_model = SlippageModel()
    initial_capital = bt_cfg.get("initial_capital", 100_000.0)
    time_budget = bt_cfg.get("time_budget_sec", 300)

    results = {}

    if split_name in ("both", "train"):
        train_bt = VectorizedBacktester(
            data=splits["train"],
            dataset_split="train",
            fee_schedule=fee_schedule,
            slippage_model=slippage_model,
            initial_capital=initial_capital,
            time_budget_sec=time_budget,
        )
        results["train"] = train_bt.run(strategy_fn, "eval", "strategy")

    if split_name in ("both", "val"):
        val_bt = VectorizedBacktester(
            data=splits["validation"],
            dataset_split="validation",
            fee_schedule=fee_schedule,
            slippage_model=slippage_model,
            initial_capital=initial_capital,
            time_budget_sec=time_budget,
        )
        results["validation"] = val_bt.run(strategy_fn, "eval", "strategy")

    if split_name == "test":
        test_bt = VectorizedBacktester(
            data=splits["test"],
            dataset_split="test",
            fee_schedule=fee_schedule,
            slippage_model=slippage_model,
            initial_capital=initial_capital,
            time_budget_sec=time_budget,
        )
        results["test"] = test_bt.run(strategy_fn, "eval", "strategy")

    # Compute OOS ratio if we have both train and val
    oos_ratio = None
    if "train" in results and "validation" in results:
        ts = results["train"]["sharpe_ratio"]
        vs = results["validation"]["sharpe_ratio"]
        safe_train = max(ts, 0.001) if ts > 0 else 0.001
        oos_ratio = vs / safe_train

    print_results(results, oos_ratio)
    return results


def print_results(results: Dict[str, Dict], oos_ratio: float | None = None):
    """Print results in a grep-friendly format (like Karpathy's train.py output)."""
    print("\n---")
    for split_name, r in results.items():
        print(f"split:            {split_name}")
        print(f"sharpe_ratio:     {r['sharpe_ratio']:.6f}")
        print(f"sortino_ratio:    {r['sortino_ratio']:.6f}")
        print(f"max_drawdown:     {r['max_drawdown']:.6f}")
        print(f"total_pnl:        {r['total_pnl']:.2f}")
        print(f"total_trades:     {r['total_trades']}")
        print(f"win_rate:         {r['win_rate']:.4f}")
        print(f"total_return:     {r['total_return']:.4f}")
        print(f"execution_sec:    {r['execution_time_sec']:.2f}")
        print(f"data_rows:        {r['data_rows']}")
        print("---")

    if oos_ratio is not None:
        print(f"oos_ratio:        {oos_ratio:.6f}")

    train_sharpe = results.get("train", {}).get("sharpe_ratio")
    val_sharpe = results.get("validation", {}).get("sharpe_ratio")
    if val_sharpe is not None:
        print(f"val_sharpe:       {val_sharpe:.6f}")
    if train_sharpe is not None:
        print(f"train_sharpe:     {train_sharpe:.6f}")
    print("---")


def main():
    parser = argparse.ArgumentParser(description="Alpha Autoresearch — Data Prep & Evaluation")
    parser.add_argument("--evaluate", action="store_true", help="Run backtest evaluation")
    parser.add_argument("--split", default="both", choices=["both", "train", "val", "test"],
                        help="Which split to evaluate (default: both = train + val)")
    parser.add_argument("--universe", type=str, default=None,
                        help="Comma-separated ticker symbols (overrides config.yaml)")
    parser.add_argument("--period-start", type=str, default=None)
    parser.add_argument("--period-end", type=str, default=None)
    args = parser.parse_args()

    if args.universe:
        universe = [s.strip() for s in args.universe.split(",")]
    else:
        universe = None

    if args.evaluate:
        evaluate(split_name=args.split)
    else:
        df = prepare_data(universe=universe, period_start=args.period_start, period_end=args.period_end)
        splits = split_data(df)
        print(f"\nData prepared: {len(df)} total rows")
        for name, sdf in splits.items():
            print(f"  {name}: {len(sdf)} rows")
        print("\nRun with --evaluate to backtest strategy.py")


if __name__ == "__main__":
    main()
