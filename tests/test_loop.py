"""
Tests for core.loop — SoloLoop keep/discard logic.

Agent is mocked so no real LLM calls are made.
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from core.agent import AgentResult
from core.loop import SoloLoop, parse_backtest_output


class TestParseBacktestOutput:
    def test_parses_sharpe(self):
        output = "train_sharpe:  0.5000\nval_sharpe:   0.3200\nmax_drawdown: -0.12\n"
        m = parse_backtest_output(output)
        assert m["val_sharpe"] == pytest.approx(0.32, abs=1e-4)
        assert m["train_sharpe"] == pytest.approx(0.5, abs=1e-4)

    def test_handles_negative_sharpe(self):
        output = "val_sharpe: -0.2728\n"
        m = parse_backtest_output(output)
        assert m["val_sharpe"] < 0

    def test_ignores_separator_lines(self):
        output = "---\nval_sharpe: 0.5\n---\n"
        m = parse_backtest_output(output)
        assert "val_sharpe" in m

    def test_empty_output_returns_empty_dict(self):
        m = parse_backtest_output("")
        assert isinstance(m, dict)

    def test_non_numeric_values_stored_as_string(self):
        output = "status: OK\n"
        m = parse_backtest_output(output)
        assert m["status"] == "OK"


class TestSoloLoopKeepDiscard:
    def _mock_agent_result(self, backtest_output: str) -> MagicMock:
        result = MagicMock(spec=AgentResult)
        result.error = None
        result.content = "Made a change to improve momentum crossover"
        result.cost_usd = 0.05
        result.elapsed_sec = 10.0
        result.tool_results = [
            {"tool": "run_backtest", "args": {}, "result_preview": backtest_output}
        ]
        return result

    def _make_loop(self, tmp_project: Path) -> SoloLoop:
        return SoloLoop(project_root=tmp_project, topic="test")

    def test_keeps_improvement(self, tmp_project):
        loop = self._make_loop(tmp_project)
        loop.best_val_sharpe = 0.0

        backtest_out = "val_sharpe:  0.50\ntrain_sharpe:  0.80\nmax_drawdown: -0.10\noos_ratio:  0.625\n"
        mock_result = self._mock_agent_result(backtest_out)

        with (
            patch("core.loop.commit_experiment", return_value="abc1234"),
            patch("core.loop.log_to_results"),
            patch("core.loop.revert_last"),
        ):
            mock_agent = MagicMock()
            mock_agent.run.return_value = mock_result
            outcome = loop.run_once(mock_agent)

        assert outcome["status"] == "keep"
        assert outcome["val_sharpe"] == pytest.approx(0.50, abs=1e-4)
        assert loop.best_val_sharpe == pytest.approx(0.50, abs=1e-4)

    def test_discards_regression(self, tmp_project):
        loop = self._make_loop(tmp_project)
        loop.best_val_sharpe = 1.0  # already high

        backtest_out = "val_sharpe:  0.30\ntrain_sharpe:  0.50\nmax_drawdown: -0.08\noos_ratio:  0.60\n"
        mock_result = self._mock_agent_result(backtest_out)

        reverted = []
        with (
            patch("core.loop.commit_experiment", return_value="def5678"),
            patch("core.loop.log_to_results"),
            patch("core.loop.revert_last", side_effect=lambda r: reverted.append(True)),
        ):
            mock_agent = MagicMock()
            mock_agent.run.return_value = mock_result
            outcome = loop.run_once(mock_agent)

        assert outcome["status"] == "discard"
        assert len(reverted) == 1  # revert was called

    def test_discards_on_oos_ratio_fail(self, tmp_project):
        loop = self._make_loop(tmp_project)
        loop.best_val_sharpe = 0.0

        # val_sharpe positive but oos_ratio below threshold
        backtest_out = "val_sharpe:  0.50\ntrain_sharpe:  2.0\nmax_drawdown: -0.08\noos_ratio:  0.10\n"
        mock_result = self._mock_agent_result(backtest_out)

        with (
            patch("core.loop.commit_experiment", return_value="ghi9012"),
            patch("core.loop.log_to_results"),
            patch("core.loop.revert_last"),
        ):
            mock_agent = MagicMock()
            mock_agent.run.return_value = mock_result
            outcome = loop.run_once(mock_agent)

        assert outcome["status"] == "discard"

    def test_handles_agent_error(self, tmp_project):
        loop = self._make_loop(tmp_project)
        error_result = MagicMock(spec=AgentResult)
        error_result.error = "CLI timed out"
        error_result.cost_usd = 0.0
        error_result.elapsed_sec = 5.0
        error_result.content = ""
        error_result.tool_results = []

        mock_agent = MagicMock()
        mock_agent.run.return_value = error_result
        outcome = loop.run_once(mock_agent)

        assert outcome["status"] == "agent_error"
        assert "CLI timed out" in outcome["error"]

    def test_falls_back_to_manual_backtest_when_agent_didnt_run(self, tmp_project):
        """If agent returned no backtest tool call, loop runs backtest manually."""
        loop = self._make_loop(tmp_project)
        loop.best_val_sharpe = 0.0

        result_no_backtest = MagicMock(spec=AgentResult)
        result_no_backtest.error = None
        result_no_backtest.content = "I modified strategy but forgot to run backtest"
        result_no_backtest.cost_usd = 0.02
        result_no_backtest.elapsed_sec = 5.0
        result_no_backtest.tool_results = []

        manual_backtest_out = "val_sharpe:  0.40\ntrain_sharpe:  0.70\nmax_drawdown: -0.05\noos_ratio:  0.57\n"

        from core.tools import ToolHandlers
        with (
            patch("core.loop.commit_experiment", return_value="jkl3456"),
            patch("core.loop.log_to_results"),
            patch("core.loop.revert_last"),
            patch.object(ToolHandlers, "run_backtest", return_value=manual_backtest_out),
        ):
            mock_agent = MagicMock()
            mock_agent.run.return_value = result_no_backtest
            outcome = loop.run_once(mock_agent)

        assert outcome["val_sharpe"] == pytest.approx(0.40, abs=1e-4)
