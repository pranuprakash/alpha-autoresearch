"""
Risk guards — overfitting prevention and statistical validity checks.

Ported from AlphaAutoResearchClaw's ProtectedKarpathyLoop.
"""

from __future__ import annotations

import logging
from typing import Dict, Any

logger = logging.getLogger("RiskGuards")


def check_oos_ratio(
    train_sharpe: float,
    val_sharpe: float,
    min_ratio: float = 0.50,
) -> Dict[str, Any]:
    """
    Out-of-sample ratio check: val_sharpe / train_sharpe must exceed threshold.
    A low ratio indicates overfitting to the training set.
    """
    safe_train = max(train_sharpe, 0.001) if train_sharpe > 0 else 0.001
    oos_ratio = val_sharpe / safe_train

    passed = oos_ratio >= min_ratio
    verdict = "PASS" if passed else "OVERFIT_DETECTED"

    logger.info(
        f"OOS check: train={train_sharpe:.4f}, val={val_sharpe:.4f}, "
        f"ratio={oos_ratio:.4f}, threshold={min_ratio}, verdict={verdict}"
    )

    return {
        "train_sharpe": train_sharpe,
        "val_sharpe": val_sharpe,
        "oos_ratio": round(oos_ratio, 4),
        "threshold": min_ratio,
        "passed": passed,
        "verdict": verdict,
    }


def check_sharpe_plausibility(sharpe: float, max_plausible: float = 6.0) -> bool:
    """Flag implausibly high Sharpe ratios (likely a bug or look-ahead bias)."""
    if abs(sharpe) > max_plausible:
        logger.warning(f"Sharpe {sharpe:.2f} exceeds plausibility threshold {max_plausible}")
        return False
    return True


def check_minimum_trades(total_trades: int, min_trades: int = 30) -> bool:
    """Ensure enough trades for statistical significance."""
    if total_trades < min_trades:
        logger.warning(f"Only {total_trades} trades (need {min_trades} for significance)")
        return False
    return True


def check_convergence(
    current_sharpe: float,
    best_sharpe: float,
    threshold: float = 0.01,
) -> bool:
    """Returns True if improvement has plateaued (< threshold relative change)."""
    if best_sharpe == 0:
        return False
    delta = abs(current_sharpe - best_sharpe) / abs(best_sharpe)
    converged = delta < threshold
    if converged:
        logger.info(f"Convergence detected: delta={delta:.4f} < threshold={threshold}")
    return converged


def full_risk_check(
    train_result: Dict[str, Any],
    val_result: Dict[str, Any],
    min_oos_ratio: float = 0.50,
    max_plausible_sharpe: float = 6.0,
    min_trades: int = 30,
) -> Dict[str, Any]:
    """Run all risk checks and return a combined verdict."""
    train_sharpe = train_result["sharpe_ratio"]
    val_sharpe = val_result["sharpe_ratio"]
    total_trades = val_result["total_trades"]

    oos = check_oos_ratio(train_sharpe, val_sharpe, min_oos_ratio)
    plausible = check_sharpe_plausibility(val_sharpe, max_plausible_sharpe)
    enough_trades = check_minimum_trades(total_trades, min_trades)

    all_passed = oos["passed"] and plausible and enough_trades

    return {
        "passed": all_passed,
        "oos_check": oos,
        "sharpe_plausible": plausible,
        "enough_trades": enough_trades,
        "val_sharpe": val_sharpe,
        "train_sharpe": train_sharpe,
    }
