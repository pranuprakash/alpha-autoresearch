"""
Multiple-testing correction — Bonferroni method.

When running many experiments, the probability of a false positive increases.
This module adjusts significance thresholds accordingly.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np
from scipy import stats

logger = logging.getLogger("MultipleTesting")


def bonferroni_correction(
    best_val_sharpe: float,
    total_iterations: int,
    significance_level: float = 0.05,
    assumed_trades: int = 100,
) -> Dict[str, Any]:
    """
    Bonferroni correction for multiple optimization iterations.

    Approximates p-value from Sharpe using:
        z = |SR| * sqrt(n_trades / 252)
        p = 2 * (1 - Phi(z))
    """
    if total_iterations == 0:
        return {"correction_method": "bonferroni", "total_iterations": 0, "significant": False}

    z_score = abs(best_val_sharpe) * np.sqrt(assumed_trades / 252)
    raw_p = 2 * (1 - stats.norm.cdf(z_score))
    adjusted_p = min(raw_p * total_iterations, 1.0)

    significant = adjusted_p < significance_level

    result = {
        "total_iterations": total_iterations,
        "best_val_sharpe": round(best_val_sharpe, 4),
        "raw_p_value": round(raw_p, 6),
        "correction_method": "bonferroni",
        "adjusted_p_value": round(adjusted_p, 6),
        "significance_level": significance_level,
        "significant_after_correction": significant,
    }

    logger.info(
        f"Bonferroni: {total_iterations} tests, "
        f"raw_p={raw_p:.6f}, adjusted_p={adjusted_p:.6f}, "
        f"significant={significant}"
    )

    return result
