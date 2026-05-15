#!/usr/bin/env python3
"""
Results dashboard — simple TUI for viewing autoresearch experiment history.

Usage:
    python scripts/dashboard.py
    python scripts/dashboard.py --last 20
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


def show_results(last_n: int = 50):
    results_path = PROJECT_ROOT / "results.tsv"
    if not results_path.exists():
        print("No results.tsv found.")
        return

    lines = results_path.read_text().strip().split("\n")
    header = lines[0]
    rows = lines[1:]

    print(f"{'='*80}")
    print(f" Alpha Autoresearch — {len(rows)} experiments")
    print(f"{'='*80}")
    print()

    keeps = [r for r in rows if "\tkeep\t" in r]
    discards = [r for r in rows if "\tdiscard\t" in r]
    crashes = [r for r in rows if "\tcrash\t" in r]

    print(f"  Kept: {len(keeps)}  |  Discarded: {len(discards)}  |  Crashed: {len(crashes)}")

    if keeps:
        best_row = max(keeps, key=lambda r: float(r.split("\t")[1]) if r.split("\t")[1].replace("-","").replace(".","").isdigit() else -999)
        parts = best_row.split("\t")
        print(f"  Best val_sharpe: {parts[1]} ({parts[0]}) — {parts[5]}")

    print()
    print(f"  {'commit':<10} {'val_sharpe':>12} {'train_sharpe':>13} {'drawdown':>10} {'status':>8}  description")
    print(f"  {'-'*10} {'-'*12} {'-'*13} {'-'*10} {'-'*8}  {'-'*30}")

    for row in rows[-last_n:]:
        parts = row.split("\t")
        if len(parts) >= 6:
            status_icon = {"keep": "+", "discard": "-", "crash": "X"}.get(parts[4], "?")
            print(f"  {parts[0]:<10} {parts[1]:>12} {parts[2]:>13} {parts[3]:>10} [{status_icon}]{parts[4]:>5}  {parts[5][:40]}")

    # Show swarm state if available
    swarm_path = PROJECT_ROOT / "artifacts" / "swarm_state.json"
    if swarm_path.exists():
        state = json.loads(swarm_path.read_text())
        print(f"\n{'='*80}")
        print(f" Swarm Agent Weights (Darwinian Evolution)")
        print(f"{'='*80}")
        weights = state.get("agent_weights", {})
        for name, w in sorted(weights.items(), key=lambda x: -x[1].get("weight", 1.0)):
            bar = "█" * int(w.get("weight", 1.0) * 10)
            sr = w.get("successful_contributions", 0)
            total = w.get("total_contributions", 0)
            print(f"  {name:<15} {w.get('weight', 1.0):>5.2f} {bar} ({sr}/{total})")


def main():
    parser = argparse.ArgumentParser(description="Alpha Autoresearch Dashboard")
    parser.add_argument("--last", type=int, default=50, help="Show last N experiments")
    args = parser.parse_args()
    show_results(args.last)


if __name__ == "__main__":
    main()
