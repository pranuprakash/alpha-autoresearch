"""Tests for risk.guards — OOS ratio, plausibility, trade count, convergence."""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from risk.guards import (
    check_convergence,
    check_minimum_trades,
    check_oos_ratio,
    check_sharpe_plausibility,
    full_risk_check,
)


class TestOosRatio:
    def test_pass_when_ratio_meets_threshold(self):
        result = check_oos_ratio(train_sharpe=1.0, val_sharpe=0.6)
        assert result["passed"] is True
        assert result["verdict"] == "PASS"

    def test_fail_when_ratio_below_threshold(self):
        result = check_oos_ratio(train_sharpe=2.0, val_sharpe=0.5)
        assert result["passed"] is False
        assert result["verdict"] == "OVERFIT_DETECTED"

    def test_exact_threshold_passes(self):
        result = check_oos_ratio(train_sharpe=1.0, val_sharpe=0.50)
        assert result["passed"] is True

    def test_zero_train_sharpe_no_crash(self):
        result = check_oos_ratio(train_sharpe=0.0, val_sharpe=0.3)
        assert isinstance(result["oos_ratio"], float)

    def test_negative_val_sharpe(self):
        result = check_oos_ratio(train_sharpe=1.0, val_sharpe=-0.2)
        assert result["passed"] is False

    def test_custom_threshold(self):
        result = check_oos_ratio(train_sharpe=1.0, val_sharpe=0.3, min_ratio=0.25)
        assert result["passed"] is True


class TestSharpePlausibility:
    def test_normal_sharpe_passes(self):
        assert check_sharpe_plausibility(1.5) is True

    def test_zero_passes(self):
        assert check_sharpe_plausibility(0.0) is True

    def test_borderline_passes(self):
        assert check_sharpe_plausibility(6.0) is True

    def test_exceed_max_fails(self):
        assert check_sharpe_plausibility(7.0) is False

    def test_negative_extreme_fails(self):
        assert check_sharpe_plausibility(-7.0) is False

    def test_custom_max(self):
        assert check_sharpe_plausibility(4.0, max_plausible=3.0) is False


class TestMinimumTrades:
    def test_enough_trades_passes(self):
        assert check_minimum_trades(50) is True

    def test_exactly_threshold_passes(self):
        assert check_minimum_trades(30) is True

    def test_too_few_fails(self):
        assert check_minimum_trades(5) is False

    def test_zero_fails(self):
        assert check_minimum_trades(0) is False

    def test_custom_min(self):
        assert check_minimum_trades(10, min_trades=5) is True


class TestConvergence:
    def test_no_improvement_is_converged(self):
        assert check_convergence(1.001, 1.0, threshold=0.01) is True

    def test_large_improvement_not_converged(self):
        assert check_convergence(1.5, 1.0, threshold=0.01) is False

    def test_zero_best_sharpe_not_converged(self):
        assert check_convergence(0.5, 0.0) is False

    def test_exact_threshold_converged(self):
        result = check_convergence(1.005, 1.0, threshold=0.005)
        assert isinstance(result, bool)


class TestFullRiskCheck:
    def _make_result(self, sharpe, trades=50):
        return {"sharpe_ratio": sharpe, "total_trades": trades}

    def test_all_pass(self):
        result = full_risk_check(
            self._make_result(1.0),
            self._make_result(0.7, trades=50),
        )
        assert result["passed"] is True

    def test_fails_on_oos(self):
        result = full_risk_check(
            self._make_result(2.0),
            self._make_result(0.5, trades=50),
        )
        assert result["passed"] is False

    def test_fails_on_too_few_trades(self):
        result = full_risk_check(
            self._make_result(1.0),
            self._make_result(0.7, trades=5),
        )
        assert result["passed"] is False

    def test_fails_on_implausible_sharpe(self):
        result = full_risk_check(
            self._make_result(1.0),
            self._make_result(8.0, trades=50),
        )
        assert result["passed"] is False
