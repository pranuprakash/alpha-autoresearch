#!/usr/bin/env python3
"""
Alpha Autoresearch — Main Entry Point.

Usage:
    python -m alpha_autoresearch.main --mode solo
    python -m alpha_autoresearch.main --mode swarm --topic "momentum optimization"
    python -m alpha_autoresearch.main --mode solo --max-iterations 10
    python -m alpha_autoresearch.main --mode swarm --max-cycles 5
"""

from __future__ import annotations

import logging
import signal
import sys
from pathlib import Path

import click

PROJECT_ROOT = Path(__file__).parent.resolve()


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)-15s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(PROJECT_ROOT / "artifacts" / "autoresearch.log"),
        ],
    )


def handle_shutdown(signum, frame):
    logging.getLogger("Main").info("Received shutdown signal. Saving state...")
    sys.exit(0)


@click.group()
def cli():
    """Alpha Autoresearch — Karpathy-style autonomous trading research."""
    pass


@cli.command()
@click.option("--mode", type=click.Choice(["solo", "swarm"]), default="solo",
              help="solo = one agent, one file. swarm = multi-agent with Darwinian evolution.")
@click.option("--topic", type=str, default="autonomous strategy optimization",
              help="Research topic / focus area")
@click.option("--max-iterations", type=int, default=None,
              help="Max iterations for solo mode (default: infinite)")
@click.option("--max-cycles", type=int, default=None,
              help="Max cycles for swarm mode (default: infinite)")
@click.option("--run-tag", type=str, default=None,
              help="Git branch tag (default: auto-generated from date)")
@click.option("--with-research", is_flag=True,
              help="Run Market Research FSM first, then feed brief into the optimization loop")
@click.option("--universe", type=str, default=None,
              help="Comma-separated tickers to scan (used with --with-research)")
@click.option("--verbose", is_flag=True, help="Debug logging")
def run(mode, topic, max_iterations, max_cycles, run_tag, with_research, universe, verbose):
    """Start the autoresearch loop."""
    (PROJECT_ROOT / "artifacts").mkdir(parents=True, exist_ok=True)
    setup_logging(verbose)
    logger = logging.getLogger("Main")

    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    logger.info(f"Alpha Autoresearch v0.2.0")
    logger.info(f"Mode: {mode} | Topic: {topic}")
    logger.info(f"Project root: {PROJECT_ROOT}")

    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")

    # Optional: run Market Research FSM first to generate a research brief
    if with_research:
        from research.fsm import MarketResearchFSM
        from research.report import format_alpha_report
        uni = [s.strip() for s in universe.split(",")] if universe else None
        logger.info("Running Market Research FSM...")
        fsm = MarketResearchFSM(
            project_root=PROJECT_ROOT,
            universe=uni,
            topic=topic,
        )
        report = fsm.run()
        print(format_alpha_report(report))
        # Update topic with the best signal if found
        if report.get("alpha_signals"):
            best = max(report["alpha_signals"], key=lambda s: s.get("confidence", 0))
            topic = f"{topic} — focus: {best.get('signal_type')} {best.get('direction')} on {best.get('ticker')}"
            logger.info(f"Updated topic from research: {topic}")

    import yaml
    config_path = PROJECT_ROOT / "config.yaml"
    config = {}
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)

    if mode == "solo":
        from core.loop import SoloLoop

        if max_iterations:
            config.setdefault("loop", {})["max_iterations"] = max_iterations

        loop = SoloLoop(
            project_root=PROJECT_ROOT,
            config=config,
            topic=topic,
        )
        loop.run(run_tag=run_tag)

    elif mode == "swarm":
        from core.swarm import SwarmOrchestrator

        swarm = SwarmOrchestrator(
            project_root=PROJECT_ROOT,
            config=config,
            topic=topic,
        )
        swarm.run(run_tag=run_tag, max_cycles=max_cycles)


@cli.command()
@click.option("--universe", type=str, default=None,
              help="Comma-separated tickers (e.g. 'SPY,QQQ,NVDA'). Default: from config.yaml")
@click.option("--topic", type=str, default="equity alpha discovery",
              help="Research topic / hypothesis to investigate")
@click.option("--resume", type=str, default=None,
              help="Resume a previous FSM run by run_id")
@click.option("--dry-run", is_flag=True,
              help="Run FSM without LLM calls (uses quantitative signals only, good for testing)")
@click.option("--verbose", is_flag=True, help="Debug logging")
def research(universe, topic, resume, dry_run, verbose):
    """Run the Market Research FSM — autonomous alpha discovery pipeline.

    Stages: UNIVERSE_SCAN → MACRO_REGIME → TECHNICAL_SCAN → SENTIMENT →
            OPTIONS_FLOW → SIGNAL_SYNTHESIS → STRATEGY_CODEGEN →
            BACKTEST_VALIDATION → ALPHA_REPORT → COMPLETE

    Outputs artifacts/research_brief.json consumed by the optimization loop.
    """
    (PROJECT_ROOT / "artifacts").mkdir(parents=True, exist_ok=True)
    setup_logging(verbose)
    logger = logging.getLogger("Research")

    symbols = [s.strip() for s in universe.split(",")] if universe else None

    from research.fsm import MarketResearchFSM
    from research.report import format_alpha_report

    fsm = MarketResearchFSM(
        project_root=PROJECT_ROOT,
        universe=symbols,
        topic=topic,
        resume_run_id=resume,
        dry_run=dry_run,
    )

    logger.info(f"Starting Market Research FSM (run_id={fsm.ctx.run_id})")
    if dry_run:
        logger.info("DRY RUN — LLM calls are skipped, using quantitative signals only")

    report = fsm.run()
    print(format_alpha_report(report))

    if report.get("validated_count", 0) > 0:
        logger.info(
            f"Run: python main.py run --mode solo --with-research to start optimization "
            f"using this research brief."
        )


@cli.command()
@click.option("--universe", type=str, default=None,
              help="Comma-separated ticker symbols (overrides config.yaml)")
def prepare(universe):
    """Download data and prepare train/val/test splits."""
    setup_logging()

    symbols = [s.strip() for s in universe.split(",")] if universe else None

    from prepare import prepare_data, split_data
    df = prepare_data(universe=symbols)
    splits = split_data(df)

    print(f"\nData prepared: {len(df)} total rows")
    for name, sdf in splits.items():
        print(f"  {name}: {len(sdf)} rows")


@cli.command()
def evaluate():
    """Run strategy.py through the backtest harness."""
    setup_logging()
    from prepare import evaluate as eval_fn
    eval_fn(split_name="both")


@cli.command()
def status():
    """Show current experiment status."""
    results_path = PROJECT_ROOT / "results.tsv"
    if not results_path.exists():
        print("No results.tsv found. Run some experiments first.")
        return

    lines = results_path.read_text().strip().split("\n")
    print(f"Total experiments: {len(lines) - 1}")

    keeps = [l for l in lines[1:] if "\tkeep\t" in l]
    discards = [l for l in lines[1:] if "\tdiscard\t" in l]
    print(f"  Kept: {len(keeps)}")
    print(f"  Discarded: {len(discards)}")

    if keeps:
        last_keep = keeps[-1].split("\t")
        print(f"  Best val_sharpe (last keep): {last_keep[1]}")

    swarm_state_path = PROJECT_ROOT / "artifacts" / "swarm_state.json"
    if swarm_state_path.exists():
        import json
        state = json.loads(swarm_state_path.read_text())
        print(f"\nSwarm state:")
        print(f"  Cycles: {state.get('cycle_count', 0)}")
        print(f"  Total cost: ${state.get('total_cost_usd', 0):.2f}")
        weights = state.get("agent_weights", {})
        if weights:
            print("  Agent weights:")
            for name, w in sorted(weights.items(), key=lambda x: -x[1].get("weight", 1.0)):
                print(f"    {name}: {w.get('weight', 1.0):.2f} "
                      f"({w.get('successful_contributions', 0)}/{w.get('total_contributions', 0)} successful)")

    print(f"\nRecent experiments:")
    for line in lines[-6:]:
        print(f"  {line}")


@cli.command()
def test_run():
    """One-shot test on the locked holdout set (SACRED — cannot be undone)."""
    setup_logging()
    logger = logging.getLogger("Main")

    click.confirm(
        "This runs the SACRED one-shot test. The test set can only be accessed ONCE. Continue?",
        abort=True,
    )

    from prepare import evaluate as eval_fn
    logger.info("Running ONE-SHOT test on locked holdout set...")
    eval_fn(split_name="test")


def main():
    cli()


if __name__ == "__main__":
    main()
