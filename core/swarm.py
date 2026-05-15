"""
Swarm Autoresearch — multi-agent orchestrator with Darwinian weight evolution.

Inspired by ATLAS (General Intelligence Capital):
- Layered specialist agents (research, strategy, optimize, audit)
- Darwinian weights: good agents get louder, bad agents get quieter
- Same keep/revert loop as solo mode, but with richer signal generation

Architecture:
    Layer 1 (Research):   3 parallel agents analyze market data
    Layer 2 (Strategy):   1 agent synthesizes research -> modifies strategy.py
    Layer 3 (Optimize):   1 agent runs Karpathy parameter optimization loop
    Layer 4 (Audit):      1 agent checks for overfitting/bias/bugs
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

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
from .loop import parse_backtest_output
from .tools import TOOL_SCHEMAS, ToolHandlers

logger = logging.getLogger("Swarm")


@dataclass
class AgentWeight:
    name: str
    weight: float = 1.0
    total_contributions: int = 0
    successful_contributions: int = 0

    def boost(self, factor: float, max_weight: float):
        self.weight = min(self.weight * factor, max_weight)

    def decay(self, factor: float, min_weight: float):
        self.weight = max(self.weight * factor, min_weight)


@dataclass
class SwarmState:
    """Persistent state for the swarm across iterations."""
    agent_weights: Dict[str, AgentWeight] = field(default_factory=dict)
    best_val_sharpe: float = -float("inf")
    cycle_count: int = 0
    total_cost_usd: float = 0.0

    def save(self, path: Path):
        data = {
            "best_val_sharpe": self.best_val_sharpe,
            "cycle_count": self.cycle_count,
            "total_cost_usd": self.total_cost_usd,
            "agent_weights": {
                name: {
                    "weight": aw.weight,
                    "total_contributions": aw.total_contributions,
                    "successful_contributions": aw.successful_contributions,
                }
                for name, aw in self.agent_weights.items()
            },
        }
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: Path) -> SwarmState:
        if not path.exists():
            return cls()
        data = json.loads(path.read_text())
        state = cls(
            best_val_sharpe=data.get("best_val_sharpe", -float("inf")),
            cycle_count=data.get("cycle_count", 0),
            total_cost_usd=data.get("total_cost_usd", 0.0),
        )
        for name, wd in data.get("agent_weights", {}).items():
            state.agent_weights[name] = AgentWeight(
                name=name,
                weight=wd.get("weight", 1.0),
                total_contributions=wd.get("total_contributions", 0),
                successful_contributions=wd.get("successful_contributions", 0),
            )
        return state


class SwarmOrchestrator:
    """
    Multi-agent trading research swarm with Darwinian weight evolution.

    Each cycle:
    1. Research agents analyze data (parallel)
    2. Strategist synthesizes research -> modifies strategy.py
    3. Optimizer fine-tunes parameters (Karpathy loop)
    4. Auditor checks for problems
    5. Keep or revert based on val_sharpe
    6. Update Darwinian weights
    """

    RESEARCH_AGENTS = ["macro", "sentiment", "technical"]

    def __init__(
        self,
        project_root: Path,
        config: Optional[Dict[str, Any]] = None,
        topic: Optional[str] = None,
    ):
        self.root = project_root.resolve()
        self.config = config or load_config(self.root / "config.yaml")
        self.topic = topic or "autonomous strategy optimization"

        self.state_path = self.root / "artifacts" / "swarm_state.json"
        self.state = SwarmState.load(self.state_path)

        loop_cfg = self.config.get("loop", {})
        self.darwinian_boost = loop_cfg.get("darwinian_boost", 1.05)
        self.darwinian_decay = loop_cfg.get("darwinian_decay", 0.95)
        self.min_weight = loop_cfg.get("min_agent_weight", 0.3)
        self.max_weight = loop_cfg.get("max_agent_weight", 2.5)
        self.karpathy_iterations = loop_cfg.get("karpathy_iterations", 5)

        risk_cfg = self.config.get("risk", {})
        self.min_oos_ratio = risk_cfg.get("min_oos_ratio", 0.50)

        for name in self.RESEARCH_AGENTS + ["strategist", "optimizer", "auditor"]:
            if name not in self.state.agent_weights:
                self.state.agent_weights[name] = AgentWeight(name=name)

    def _load_prompt(self, prompt_name: str) -> str:
        """Load a prompt from the prompts/ directory."""
        prompt_path = self.root / "prompts" / f"{prompt_name}.md"
        if prompt_path.exists():
            return prompt_path.read_text()
        return f"You are the {prompt_name} agent in a trading research swarm."

    def _build_research_agent(self, specialty: str) -> Agent:
        model = self.config.get("models", {}).get("researcher", "google/gemini-2.5-flash")
        prompt = self._load_prompt("researcher")
        prompt += f"\n\nYour specialty: {specialty.upper()} analysis."
        prompt += f"\nYour current Darwinian weight: {self.state.agent_weights.get(specialty, AgentWeight(specialty)).weight:.2f}"

        tools = ToolHandlers(self.root)
        return Agent(
            name=f"researcher_{specialty}",
            model=model,
            system_prompt=prompt,
            tools=TOOL_SCHEMAS,
            tool_handlers=tools.get_handlers(),
        )

    def _build_strategist(self) -> Agent:
        model = self.config.get("models", {}).get("strategist", "anthropic/claude-sonnet-4-20250514")
        prompt = self._load_prompt("strategist")
        tools = ToolHandlers(self.root)
        return Agent(
            name="strategist",
            model=model,
            system_prompt=prompt,
            tools=TOOL_SCHEMAS,
            tool_handlers=tools.get_handlers(),
        )

    def _build_optimizer(self) -> Agent:
        model = self.config.get("models", {}).get("optimizer", "anthropic/claude-sonnet-4-20250514")
        prompt = self._load_prompt("optimizer")
        tools = ToolHandlers(self.root)
        return Agent(
            name="optimizer",
            model=model,
            system_prompt=prompt,
            tools=TOOL_SCHEMAS,
            tool_handlers=tools.get_handlers(),
        )

    def _build_auditor(self) -> Agent:
        model = self.config.get("models", {}).get("auditor", "openai/gpt-4o")
        prompt = self._load_prompt("auditor")
        tools = ToolHandlers(self.root)
        return Agent(
            name="auditor",
            model=model,
            system_prompt=prompt,
            tools=[TOOL_SCHEMAS[0], TOOL_SCHEMAS[3]],  # read_file + list_files only
            tool_handlers=ToolHandlers(self.root).get_handlers(),
        )

    def _load_research_brief_context(self) -> str:
        """Load and format the research brief for injection into agent prompts."""
        brief_path = self.root / "artifacts" / "research_brief.json"
        if not brief_path.exists():
            return ""
        try:
            from research.report import brief_to_context_string, load_brief
            brief = load_brief(self.root)
            return brief_to_context_string(brief)
        except Exception:
            return ""

    def _run_research_phase(self) -> Dict[str, Any]:
        """Layer 1: Run research agents (sequentially for simplicity, could be parallelized)."""
        logger.info("=== LAYER 1: RESEARCH ===")
        research_results = {}

        research_brief_ctx = self._load_research_brief_context()
        if research_brief_ctx:
            logger.info("  Injecting pre-computed research brief into agent context")

        for specialty in self.RESEARCH_AGENTS:
            agent = self._build_research_agent(specialty)
            weight = self.state.agent_weights[specialty].weight

            message = (
                f"Analyze the market data for: {self.topic}\n"
                f"Read strategy.py and results.tsv first for context.\n"
                f"Then read the data files in data/cache/ if available.\n"
                f"Your Darwinian weight: {weight:.2f} (higher = more trusted)\n"
            )
            if research_brief_ctx:
                message += f"\nPRE-COMPUTED RESEARCH (use as starting point):\n{research_brief_ctx}\n"
            message += "\nProvide your research brief as described in your instructions."

            result = agent.run(message)
            self.state.total_cost_usd += result.cost_usd

            research_results[specialty] = {
                "content": result.content,
                "weight": weight,
                "cost_usd": result.cost_usd,
            }

            logger.info(f"  {specialty}: {len(result.content)} chars, ${result.cost_usd:.3f}")

        return research_results

    def _run_strategy_phase(self, research: Dict[str, Any]) -> AgentResult:
        """Layer 2: Strategist synthesizes research into strategy.py modifications."""
        logger.info("=== LAYER 2: STRATEGY ===")
        strategist = self._build_strategist()

        research_summary = "\n\n".join([
            f"## {name.upper()} (weight: {r['weight']:.2f})\n{r['content']}"
            for name, r in research.items()
        ])

        message = (
            f"Topic: {self.topic}\n"
            f"Best val_sharpe so far: {self.state.best_val_sharpe:.4f}\n\n"
            f"=== RESEARCH INPUTS ===\n{research_summary}\n"
            f"=== END RESEARCH ===\n\n"
            f"Based on the research, modify strategy.py to improve val_sharpe.\n"
            f"Read the current strategy.py first, then write your improvements.\n"
            f"Then run the backtest to see results."
        )

        result = strategist.run(message)
        self.state.total_cost_usd += result.cost_usd
        logger.info(f"  Strategist: {result.tool_calls_made} tool calls, ${result.cost_usd:.3f}")
        return result

    def _run_optimize_phase(self) -> List[Dict[str, Any]]:
        """Layer 3: Optimizer runs Karpathy iterations on parameters."""
        logger.info("=== LAYER 3: OPTIMIZE ===")
        optimizer = self._build_optimizer()
        results = []

        for i in range(self.karpathy_iterations):
            message = (
                f"Optimization iteration {i+1}/{self.karpathy_iterations}\n"
                f"Best val_sharpe: {self.state.best_val_sharpe:.4f}\n"
                f"Read strategy.py, identify a parameter to tune, modify it, "
                f"and run the backtest."
            )

            result = optimizer.run(message)
            self.state.total_cost_usd += result.cost_usd

            backtest_output = None
            for tr in result.tool_results:
                if tr["tool"] == "run_backtest":
                    backtest_output = tr["result_preview"]
                    break

            if backtest_output:
                metrics = parse_backtest_output(backtest_output)
                val_sharpe = metrics.get("val_sharpe", 0.0)
                if isinstance(val_sharpe, str):
                    val_sharpe = 0.0
            else:
                val_sharpe = 0.0

            iteration_result = {
                "iteration": i + 1,
                "val_sharpe": val_sharpe,
                "cost_usd": result.cost_usd,
                "description": result.content[:100] if result.content else "",
            }
            results.append(iteration_result)

            if val_sharpe > self.state.best_val_sharpe:
                commit_experiment(self.root, f"optimize_iter_{i+1}: {result.content[:60]}")
                self.state.best_val_sharpe = val_sharpe
                logger.info(f"  Iter {i+1}: KEEP val_sharpe={val_sharpe:.4f}")
            else:
                revert_last(self.root)
                logger.info(f"  Iter {i+1}: DISCARD val_sharpe={val_sharpe:.4f}")

        return results

    def _run_audit_phase(self) -> Dict[str, Any]:
        """Layer 4: Auditor checks the final strategy."""
        logger.info("=== LAYER 4: AUDIT ===")
        auditor = self._build_auditor()

        results_tsv = ""
        results_path = self.root / "results.tsv"
        if results_path.exists():
            results_tsv = results_path.read_text()

        message = (
            f"Audit the current strategy.py for the topic: {self.topic}\n"
            f"Read strategy.py and check for:\n"
            f"1. Look-ahead bias\n"
            f"2. Overfitting signals\n"
            f"3. Implementation bugs\n"
            f"4. Statistical significance\n\n"
            f"Results history:\n{results_tsv}\n\n"
            f"Provide your verdict: APPROVED, REVIEW_REQUIRED, or REJECTED."
        )

        result = auditor.run(message)
        self.state.total_cost_usd += result.cost_usd

        content = result.content.upper()
        if "REJECTED" in content:
            verdict = "REJECTED"
        elif "APPROVED" in content:
            verdict = "APPROVED"
        else:
            verdict = "REVIEW_REQUIRED"

        logger.info(f"  Auditor verdict: {verdict}, ${result.cost_usd:.3f}")

        return {
            "verdict": verdict,
            "content": result.content,
            "cost_usd": result.cost_usd,
        }

    def _update_darwinian_weights(self, success: bool, research: Dict[str, Any]):
        """Update agent weights based on cycle outcome."""
        for name in self.RESEARCH_AGENTS:
            aw = self.state.agent_weights[name]
            aw.total_contributions += 1
            if success:
                aw.successful_contributions += 1
                aw.boost(self.darwinian_boost, self.max_weight)
            else:
                aw.decay(self.darwinian_decay, self.min_weight)

        for name in ["strategist", "optimizer", "auditor"]:
            aw = self.state.agent_weights[name]
            aw.total_contributions += 1
            if success:
                aw.successful_contributions += 1
                aw.boost(self.darwinian_boost, self.max_weight)
            else:
                aw.decay(self.darwinian_decay, self.min_weight)

    def run_cycle(self) -> Dict[str, Any]:
        """Run one complete swarm cycle (research -> strategy -> optimize -> audit)."""
        self.state.cycle_count += 1
        cycle = self.state.cycle_count
        start = time.monotonic()

        logger.info(f"\n{'='*60}")
        logger.info(f"SWARM CYCLE {cycle}")
        logger.info(f"{'='*60}")

        # Save strategy.py before modifications
        strategy_path = self.root / "strategy.py"
        original_strategy = strategy_path.read_text() if strategy_path.exists() else ""

        # Layer 1: Research
        research = self._run_research_phase()

        # Layer 2: Strategy
        strategy_result = self._run_strategy_phase(research)

        # Commit the strategy change
        description = strategy_result.content[:80] if strategy_result.content else f"swarm_cycle_{cycle}"
        commit_hash = commit_experiment(self.root, description) or "0000000"

        # Get metrics after strategy phase
        tools = ToolHandlers(self.root)
        backtest_output = tools.run_backtest()
        metrics = parse_backtest_output(backtest_output)
        post_strategy_sharpe = metrics.get("val_sharpe", 0.0)
        if isinstance(post_strategy_sharpe, str):
            post_strategy_sharpe = 0.0

        # Layer 3: Optimize (Karpathy iterations)
        optimize_results = self._run_optimize_phase()

        # Get final metrics
        final_output = tools.run_backtest()
        final_metrics = parse_backtest_output(final_output)
        final_val_sharpe = final_metrics.get("val_sharpe", 0.0)
        final_train_sharpe = final_metrics.get("train_sharpe", 0.0)
        final_max_dd = final_metrics.get("max_drawdown", 0.0)
        for k in ["val_sharpe", "train_sharpe", "max_drawdown"]:
            v = final_metrics.get(k, 0.0)
            if isinstance(v, str):
                final_metrics[k] = 0.0

        # Layer 4: Audit
        audit = self._run_audit_phase()

        # Decision: keep or revert
        improved = final_val_sharpe > self.state.best_val_sharpe
        approved = audit["verdict"] != "REJECTED"
        success = improved and approved

        if success:
            self.state.best_val_sharpe = final_val_sharpe
            status = "keep"
            logger.info(f"CYCLE {cycle}: KEEP — val_sharpe={final_val_sharpe:.4f}")
        else:
            # Revert to original strategy
            strategy_path.write_text(original_strategy)
            commit_experiment(self.root, f"revert_cycle_{cycle}")
            status = "discard"
            reasons = []
            if not improved:
                reasons.append("no improvement")
            if not approved:
                reasons.append(f"audit: {audit['verdict']}")
            logger.info(f"CYCLE {cycle}: DISCARD — {'; '.join(reasons)}")

        # Update Darwinian weights
        self._update_darwinian_weights(success, research)

        # Log results
        if isinstance(final_val_sharpe, (int, float)):
            log_to_results(
                self.root, commit_hash,
                float(final_val_sharpe),
                float(final_train_sharpe) if isinstance(final_train_sharpe, (int, float)) else 0.0,
                float(final_max_dd) if isinstance(final_max_dd, (int, float)) else 0.0,
                status, description[:120],
            )

        # Save swarm state
        (self.root / "artifacts").mkdir(parents=True, exist_ok=True)
        self.state.save(self.state_path)

        elapsed = time.monotonic() - start

        return {
            "cycle": cycle,
            "status": status,
            "val_sharpe": final_val_sharpe,
            "audit_verdict": audit["verdict"],
            "optimize_iterations": len(optimize_results),
            "cost_usd": sum(r.get("cost_usd", 0) for r in [
                {"cost_usd": sum(r["cost_usd"] for r in research.values())},
                {"cost_usd": strategy_result.cost_usd},
                {"cost_usd": sum(r["cost_usd"] for r in optimize_results)},
                {"cost_usd": audit["cost_usd"]},
            ]),
            "elapsed_sec": elapsed,
            "agent_weights": {
                name: round(aw.weight, 2)
                for name, aw in self.state.agent_weights.items()
            },
        }

    def run(self, run_tag: Optional[str] = None, max_cycles: Optional[int] = None):
        """
        Main swarm loop — runs cycles until interrupted.
        """
        from dotenv import load_dotenv
        load_dotenv(self.root / ".env")

        if git_available(self.root):
            tag = run_tag or datetime.now().strftime("%b%d").lower()
            branch_name = f"autoresearch-swarm/{tag}"
            create_branch(self.root, branch_name)

        init_results_tsv(self.root)

        logger.info(f"Starting swarm autoresearch: topic='{self.topic}'")
        logger.info(f"Agents: {', '.join(self.RESEARCH_AGENTS)} + strategist + optimizer + auditor")
        logger.info("NEVER STOP — loop runs until manually interrupted")

        # Run baseline
        tools = ToolHandlers(self.root)
        baseline_output = tools.run_backtest()
        baseline_metrics = parse_backtest_output(baseline_output)
        baseline_val = baseline_metrics.get("val_sharpe", 0.0)
        if isinstance(baseline_val, str):
            baseline_val = 0.0
        self.state.best_val_sharpe = baseline_val

        commit_hash = current_commit(self.root) if git_available(self.root) else "baseline"
        baseline_train = baseline_metrics.get("train_sharpe", 0.0)
        if isinstance(baseline_train, str):
            baseline_train = 0.0
        baseline_dd = baseline_metrics.get("max_drawdown", 0.0)
        if isinstance(baseline_dd, str):
            baseline_dd = 0.0
        log_to_results(self.root, commit_hash, baseline_val, baseline_train, baseline_dd, "keep", "baseline")
        logger.info(f"Baseline: val_sharpe={baseline_val:.4f}")

        cycle = 0
        while True:
            cycle += 1
            if max_cycles and cycle > max_cycles:
                logger.info(f"Reached max_cycles ({max_cycles}). Stopping.")
                break

            try:
                result = self.run_cycle()
                logger.info(
                    f"[Cycle {result['cycle']}] {result['status']} | "
                    f"val_sharpe={result['val_sharpe']:.4f} | "
                    f"audit={result['audit_verdict']} | "
                    f"${result['cost_usd']:.2f} | "
                    f"{result['elapsed_sec']:.0f}s"
                )

                weights_str = ", ".join(
                    f"{name}={w}" for name, w in result["agent_weights"].items()
                )
                logger.info(f"  Weights: {weights_str}")

            except KeyboardInterrupt:
                logger.info("Interrupted by user. Saving state and stopping.")
                self.state.save(self.state_path)
                break
            except Exception as e:
                logger.error(f"Cycle {cycle} crashed: {e}", exc_info=True)
                logger.info("Saving state and continuing to next cycle...")
                self.state.save(self.state_path)
                continue

        logger.info(f"\nFinal best val_sharpe: {self.state.best_val_sharpe:.4f}")
        logger.info(f"Total cycles: {self.state.cycle_count}")
        logger.info(f"Total cost: ${self.state.total_cost_usd:.2f}")
