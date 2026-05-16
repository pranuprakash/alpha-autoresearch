"""
Solo Autoresearch Loop — Karpathy-pure.

One agent. One file. One metric. Keep or revert. NEVER STOP.

Mirrors Karpathy's program.md loop:
1. Agent reads program.md + strategy.py + results.tsv
2. Agent proposes modification to strategy.py
3. git commit
4. Run backtest (prepare.py --evaluate)
5. If val_sharpe improved AND risk checks pass: KEEP
6. If not: git reset --hard HEAD~1
7. Log to results.tsv
8. Repeat
"""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from .agent import Agent, AgentResult, load_config
from .git_ops import (
    commit_experiment,
    create_branch,
    current_commit,
    git_available,
    init_results_tsv,
    log_to_results,
    revert_last,
)
from .tools import TOOL_SCHEMAS, ToolHandlers

logger = logging.getLogger("SoloLoop")


def parse_backtest_output(output: str) -> Dict[str, float]:
    """Extract metrics from prepare.py --evaluate output (grep-friendly format)."""
    metrics = {}
    for line in output.split("\n"):
        line = line.strip()
        if ":" in line and not line.startswith("---"):
            key, _, value = line.partition(":")
            key = key.strip()
            value = value.strip()
            try:
                metrics[key] = float(value)
            except ValueError:
                metrics[key] = value
    return metrics


class SoloLoop:
    """
    The Karpathy-pure autoresearch loop for trading.

    Creates an agent, gives it tools to read/write files and run backtests,
    and loops: propose change -> backtest -> keep/revert -> log.
    """

    def __init__(
        self,
        project_root: Path,
        config: Optional[Dict[str, Any]] = None,
        topic: Optional[str] = None,
    ):
        self.root = project_root.resolve()
        self.config = config or load_config(self.root / "config.yaml")
        self.topic = topic or "autonomous strategy optimization"

        risk_cfg = self.config.get("risk", {})
        self.min_oos_ratio = risk_cfg.get("min_oos_ratio", 0.50)
        self.max_plausible_sharpe = risk_cfg.get("max_plausible_sharpe", 6.0)

        loop_cfg = self.config.get("loop", {})
        self.max_iterations = loop_cfg.get("max_iterations")  # None = infinite
        self.convergence_threshold = loop_cfg.get("convergence_threshold", 0.01)

        self.best_val_sharpe = -float("inf")
        self.iteration_count = 0

    def _build_agent(self) -> Agent:
        """Create the solo agent with tools and system prompt."""
        model = self.config.get("models", {}).get("solo_agent", "claude-cli/claude-sonnet-4-6")

        program_path = self.root / "program.md"
        if program_path.exists():
            system_prompt = program_path.read_text()
        else:
            system_prompt = "You are an autonomous trading strategy researcher."

        tools = ToolHandlers(self.root)

        return Agent(
            name="SoloResearcher",
            model=model,
            system_prompt=system_prompt,
            tools=TOOL_SCHEMAS,
            tool_handlers=tools.get_handlers(),
        )

    def _build_iteration_message(self) -> str:
        """Build the message sent to the agent each iteration."""
        parts = [f"Topic: {self.topic}"]
        parts.append(f"Iteration: {self.iteration_count + 1}")
        parts.append(f"Best val_sharpe so far: {self.best_val_sharpe:.6f}")

        results_path = self.root / "results.tsv"
        if results_path.exists():
            results_content = results_path.read_text()
            lines = results_content.strip().split("\n")
            if len(lines) > 1:
                recent = "\n".join(lines[:1] + lines[-10:])
                parts.append(f"\nRecent results (last 10):\n{recent}")

        parts.append(
            "\nYour task:\n"
            "1. Read the current strategy.py using the read_file tool\n"
            "2. Analyze the code and results history\n"
            "3. Propose a specific modification to improve val_sharpe\n"
            "4. Write your modified strategy.py using the write_file tool\n"
            "5. Run the backtest using the run_backtest tool\n"
            "6. Report what you changed and the results\n"
            "\nIMPORTANT: Always write the COMPLETE strategy.py file. "
            "The file must define generate_signals(df) -> pd.Series. "
            "Be creative — change indicators, parameters, logic, add new signals. "
            "Simpler is better when results are similar."
        )

        return "\n".join(parts)

    def run_once(self, agent: Agent) -> Dict[str, Any]:
        """Run a single experiment iteration."""
        self.iteration_count += 1
        iteration = self.iteration_count

        logger.info(f"{'='*60}")
        logger.info(f"Iteration {iteration} — best_val_sharpe={self.best_val_sharpe:.4f}")
        logger.info(f"{'='*60}")

        message = self._build_iteration_message()
        result = agent.run(message)

        if result.error:
            logger.error(f"Agent error: {result.error}")
            return {"status": "agent_error", "error": result.error, "iteration": iteration}

        logger.info(
            f"Agent finished: {result.tool_calls_made} tool calls, "
            f"${result.cost_usd:.4f}, {result.elapsed_sec:.1f}s"
        )

        # The agent should have written strategy.py and run the backtest via tools.
        # Parse the backtest results from the agent's tool calls.
        backtest_output = None
        for tr in result.tool_results:
            if tr["tool"] == "run_backtest":
                backtest_output = tr["result_preview"]
                break

        if backtest_output is None:
            logger.warning("Agent did not run backtest — running manually")
            tools = ToolHandlers(self.root)
            backtest_output = tools.run_backtest()

        metrics = parse_backtest_output(backtest_output)
        val_sharpe = metrics.get("val_sharpe", 0.0)
        train_sharpe = metrics.get("train_sharpe", 0.0)
        max_drawdown = metrics.get("max_drawdown", 0.0)
        oos_ratio = metrics.get("oos_ratio", 0.0)

        if isinstance(val_sharpe, str):
            val_sharpe = 0.0
        if isinstance(train_sharpe, str):
            train_sharpe = 0.0
        if isinstance(max_drawdown, str):
            max_drawdown = 0.0
        if isinstance(oos_ratio, str):
            oos_ratio = 0.0

        description = result.content[:200] if result.content else f"iteration_{iteration}"
        description = description.replace("\t", " ").replace("\n", " ").strip()
        if not description:
            description = f"iteration_{iteration}"

        improved = val_sharpe > self.best_val_sharpe
        risk_ok = oos_ratio >= self.min_oos_ratio if oos_ratio > 0 else True
        plausible = abs(val_sharpe) <= self.max_plausible_sharpe

        commit_hash = commit_experiment(self.root, description[:80]) or "0000000"

        if improved and risk_ok and plausible:
            status = "keep"
            self.best_val_sharpe = val_sharpe
            logger.info(f"KEEP — val_sharpe improved to {val_sharpe:.4f}")
        else:
            status = "discard"
            revert_last(self.root)
            reasons = []
            if not improved:
                reasons.append(f"no improvement ({val_sharpe:.4f} <= {self.best_val_sharpe:.4f})")
            if not risk_ok:
                reasons.append(f"OOS ratio {oos_ratio:.4f} < {self.min_oos_ratio}")
            if not plausible:
                reasons.append(f"implausible Sharpe {val_sharpe:.4f}")
            logger.info(f"DISCARD — {'; '.join(reasons)}")

        log_to_results(
            self.root, commit_hash, val_sharpe, train_sharpe, max_drawdown,
            status, description[:120],
        )

        return {
            "iteration": iteration,
            "status": status,
            "val_sharpe": val_sharpe,
            "train_sharpe": train_sharpe,
            "max_drawdown": max_drawdown,
            "oos_ratio": oos_ratio,
            "commit": commit_hash,
            "description": description[:120],
            "cost_usd": result.cost_usd,
            "elapsed_sec": result.elapsed_sec,
        }

    def _check_convergence(self) -> bool:
        """Check if we should stop due to convergence."""
        results_path = self.root / "results.tsv"
        if not results_path.exists():
            return False

        lines = results_path.read_text().strip().split("\n")[1:]  # skip header
        if len(lines) < 5:
            return False

        recent_keeps = [l for l in lines[-10:] if "\tkeep\t" in l]
        if len(recent_keeps) == 0 and len(lines) > 20:
            logger.info("No keeps in last 10 experiments — possible convergence")
            return True

        return False

    def run(self, run_tag: Optional[str] = None):
        """
        Main loop — runs until interrupted or max_iterations.
        NEVER STOP unless explicitly told to.
        """
        from dotenv import load_dotenv
        load_dotenv(self.root / ".env")

        from core.claude_cli import verify_claude_cli
        verify_claude_cli()

        if git_available(self.root):
            tag = run_tag or datetime.now().strftime("%b%d").lower()
            branch_name = f"autoresearch/{tag}"
            create_branch(self.root, branch_name)
        else:
            logger.warning("Git not available — running without version control")

        init_results_tsv(self.root)
        agent = self._build_agent()

        logger.info(f"Starting solo autoresearch loop: topic='{self.topic}'")
        logger.info(f"Model: {agent.model}")
        logger.info("NEVER STOP — loop runs until manually interrupted (Ctrl+C)")

        # Run baseline first
        logger.info("Running baseline backtest...")
        tools = ToolHandlers(self.root)
        baseline_output = tools.run_backtest()
        baseline_metrics = parse_backtest_output(baseline_output)
        baseline_val = baseline_metrics.get("val_sharpe", 0.0)
        if isinstance(baseline_val, str):
            baseline_val = 0.0
        self.best_val_sharpe = baseline_val

        baseline_train = baseline_metrics.get("train_sharpe", 0.0)
        if isinstance(baseline_train, str):
            baseline_train = 0.0
        baseline_dd = baseline_metrics.get("max_drawdown", 0.0)
        if isinstance(baseline_dd, str):
            baseline_dd = 0.0

        commit_hash = current_commit(self.root) if git_available(self.root) else "baseline"
        log_to_results(
            self.root, commit_hash, baseline_val, baseline_train,
            baseline_dd, "keep", "baseline",
        )
        logger.info(f"Baseline: val_sharpe={baseline_val:.4f}")

        iteration = 0
        while True:
            iteration += 1

            if self.max_iterations and iteration > self.max_iterations:
                logger.info(f"Reached max_iterations ({self.max_iterations}). Stopping.")
                break

            try:
                result = self.run_once(agent)
                logger.info(
                    f"[{result['iteration']}] {result['status']} | "
                    f"val_sharpe={result['val_sharpe']:.4f} | "
                    f"${result['cost_usd']:.3f} | "
                    f"{result['elapsed_sec']:.0f}s"
                )
            except KeyboardInterrupt:
                logger.info("Interrupted by user. Stopping gracefully.")
                break
            except Exception as e:
                logger.error(f"Iteration {iteration} crashed: {e}", exc_info=True)
                logger.info("Continuing to next iteration...")
                continue

        logger.info(f"\nFinal best val_sharpe: {self.best_val_sharpe:.4f}")
        logger.info(f"Total iterations: {self.iteration_count}")
        logger.info("Results logged to results.tsv")
