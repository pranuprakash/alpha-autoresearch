"""
Risk Audit — adversarial checks for strategy quality.

Ported from AlphaAutoResearchClaw's risk_audit.py.
Checks: look-ahead bias, overfitting signals, statistical significance.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger("RiskAudit")

LOOKAHEAD_PATTERNS = [
    r"\.shift\s*\(\s*-\d+",       # negative shift = future data
    r"merge_asof.*direction\s*=\s*['\"]forward",
    r"\.iloc\s*\[\s*\d+\s*:\s*\].*\.iloc\s*\[\s*-",  # suspicious indexing
]


def check_lookahead_bias(strategy_path: Path) -> Dict[str, Any]:
    """Static analysis of strategy.py for look-ahead bias patterns."""
    if not strategy_path.exists():
        return {"passed": False, "error": "strategy.py not found"}

    code = strategy_path.read_text()
    violations = []

    for pattern in LOOKAHEAD_PATTERNS:
        matches = re.findall(pattern, code)
        if matches:
            violations.append({"pattern": pattern, "matches": matches})

    passed = len(violations) == 0
    if not passed:
        logger.warning(f"Look-ahead bias detected: {len(violations)} pattern(s)")

    return {
        "check": "lookahead_bias",
        "passed": passed,
        "violations": violations,
    }


def check_overfitting_signals(
    train_sharpe: float,
    val_sharpe: float,
    total_iterations: int,
) -> Dict[str, Any]:
    """Heuristic checks for overfitting."""
    warnings: List[str] = []

    if train_sharpe > 0 and val_sharpe < 0:
        warnings.append("Train positive but validation negative — classic overfit")

    if train_sharpe > 3.0 and val_sharpe < 1.0:
        warnings.append(f"Train Sharpe ({train_sharpe:.2f}) vastly exceeds val ({val_sharpe:.2f})")

    if total_iterations > 10:
        warnings.append(f"High iteration count ({total_iterations}) increases multiple-testing risk")

    return {
        "check": "overfitting_signals",
        "passed": len(warnings) == 0,
        "warnings": warnings,
    }


def run_full_audit(
    strategy_path: Path,
    train_sharpe: float,
    val_sharpe: float,
    total_iterations: int,
) -> Dict[str, Any]:
    """Run all audit checks and compute a safety score."""
    lookahead = check_lookahead_bias(strategy_path)
    overfit = check_overfitting_signals(train_sharpe, val_sharpe, total_iterations)

    checks = [lookahead, overfit]
    passed_count = sum(1 for c in checks if c["passed"])
    total_checks = len(checks)
    safety_score = int(100 * passed_count / total_checks)

    all_passed = all(c["passed"] for c in checks)

    return {
        "all_passed": all_passed,
        "safety_score": safety_score,
        "checks": checks,
        "verdict": "SAFE" if all_passed else "REVIEW_REQUIRED",
    }
