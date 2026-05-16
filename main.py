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

# When invoked as CLI entry point (alpha-research), alpha_autoresearch/ is not in sys.path.
# This ensures submodule imports (core.loop, research.fsm, etc.) resolve correctly.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


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


@cli.command()
@click.option("--universe", type=str, default=None,
              help="Comma-separated tickers (e.g. 'NVDA,META,SPY'). Reads brief if omitted.")
@click.option("--brief", type=click.Path(exists=True), default=None,
              help="Path to research_brief.json. Defaults to artifacts/research_brief.json.")
@click.option("--portfolio-value", type=float, default=100_000,
              help="Total portfolio value for position sizing (default: $100,000)")
@click.option("--risk-pct", type=float, default=0.02,
              help="Max risk per play as fraction of portfolio (default: 0.02 = 2%%)")
@click.option("--no-options", is_flag=True,
              help="Generate equity plays instead of options plays")
@click.option("--output", type=click.Path(), default=None,
              help="Save PlayBook JSON to this path")
@click.option("--verbose", is_flag=True)
def plays(universe, brief, portfolio_value, risk_pct, no_options, output, verbose):
    """Generate specific trade tickets from a research brief.

    Reads artifacts/research_brief.json (or --brief path) and converts
    alpha signals into exact trade tickets with entry/limit/target/stop/sizing.

    Optionally run the research FSM first:
        python main.py research --universe "NVDA,META" --dry-run
        python main.py plays
    """
    (PROJECT_ROOT / "artifacts").mkdir(parents=True, exist_ok=True)
    setup_logging(verbose)
    logger = logging.getLogger("Plays")

    from plays.generator import PlayGenerator, format_playbook

    brief_path = PROJECT_ROOT / "artifacts" / "research_brief.json"
    if brief:
        brief_path = Path(brief)

    if not brief_path.exists():
        if universe:
            logger.info("No brief found — running Research FSM first...")
            from research.fsm import MarketResearchFSM
            symbols = [s.strip() for s in universe.split(",")]
            fsm = MarketResearchFSM(project_root=PROJECT_ROOT, universe=symbols, dry_run=True)
            fsm.run()
        else:
            print("No research_brief.json found. Run:\n"
                  "  python main.py research --universe 'NVDA,META,SPY' --dry-run\n"
                  "Then re-run: python main.py plays")
            return

    gen = PlayGenerator(
        portfolio_value=portfolio_value,
        risk_pct=risk_pct,
        prefer_options=not no_options,
    )

    logger.info(f"Generating plays from {brief_path}")
    book = gen.generate(brief_path=brief_path)
    print(format_playbook(book))

    if output:
        import json as _json
        Path(output).write_text(_json.dumps(book.to_dict(), indent=2))
        logger.info(f"PlayBook saved to {output}")
    else:
        out = PROJECT_ROOT / "artifacts" / "playbook.json"
        import json as _json
        out.write_text(_json.dumps(book.to_dict(), indent=2))
        logger.info(f"PlayBook saved to {out}")


@cli.command()
@click.option("--file", "portfolio_file", type=click.Path(exists=True), required=True,
              help="Path to portfolio JSON (see artifacts/sample_portfolio.json for format)")
@click.option("--portfolio-value", type=float, default=None,
              help="Override computed portfolio value (e.g. actual account value)")
@click.option("--output", type=click.Path(), default=None,
              help="Save enriched portfolio + report JSON to this path")
@click.option("--verbose", is_flag=True)
def portfolio(portfolio_file, portfolio_value, output, verbose):
    """Analyze your current portfolio: live prices, Greeks, P&L, risk metrics.

    Input JSON format (save as my_portfolio.json):
    {
      "cash": 10000,
      "positions": [
        {"asset_type": "equity", "symbol": "NVDA", "shares": 10, "cost_basis": 850.0},
        {"asset_type": "option", "symbol": "META", "option_type": "call",
         "strike": 700, "expiry": "2026-10-16", "quantity": 1, "cost_basis": 31.75}
      ]
    }
    """
    setup_logging(verbose)
    from portfolio.engine import PortfolioEngine
    from portfolio.recommender import ActionRecommender
    from portfolio.reporter import format_portfolio_report

    eng = PortfolioEngine(portfolio_value_override=portfolio_value)
    port = eng.load(portfolio_path=Path(portfolio_file))
    enriched = eng.enrich(port)

    rec = ActionRecommender()
    report = rec.recommend(enriched)

    print(format_portfolio_report(enriched, report))

    if output:
        import json as _json
        Path(output).write_text(_json.dumps(report.to_dict(), indent=2))


@cli.command()
@click.option("--file", "portfolio_file", type=click.Path(exists=True), required=True,
              help="Path to portfolio JSON")
@click.option("--universe", type=str, default=None,
              help="Comma-separated tickers to research for new plays")
@click.option("--brief", type=click.Path(), default=None,
              help="Path to existing research_brief.json (skip FSM if provided)")
@click.option("--portfolio-value", type=float, default=None,
              help="Override total portfolio value")
@click.option("--risk-pct", type=float, default=0.02,
              help="Max risk per new play (default: 2%%)")
@click.option("--no-options", is_flag=True, help="Equity plays only, no options")
@click.option("--dry-run", is_flag=True, help="Run FSM in dry-run mode (no LLM)")
@click.option("--output", type=click.Path(), default=None,
              help="Save full recommendation JSON to this path")
@click.option("--verbose", is_flag=True)
def recommend(portfolio_file, universe, brief, portfolio_value, risk_pct,
              no_options, dry_run, output, verbose):
    """Full pipeline: analyze portfolio + generate new plays + produce recommendations.

    This is the end-to-end command:
      1. Loads and enriches your portfolio (live prices + Greeks)
      2. Runs Market Research FSM on --universe (or uses existing --brief)
      3. Generates new trade tickets
      4. Filters plays by portfolio constraints (concentration, cash, risk budget)
      5. Prints: position actions + new trades + capital summary

    Example:
        python main.py recommend --file my_portfolio.json --universe "NVDA,META,SPY"
    """
    (PROJECT_ROOT / "artifacts").mkdir(parents=True, exist_ok=True)
    setup_logging(verbose)
    logger = logging.getLogger("Recommend")

    from portfolio.engine import PortfolioEngine
    from portfolio.recommender import ActionRecommender
    from portfolio.reporter import format_portfolio_report
    from plays.generator import PlayGenerator

    # 1. Load + enrich portfolio
    eng = PortfolioEngine(portfolio_value_override=portfolio_value)
    port = eng.load(portfolio_path=Path(portfolio_file))
    logger.info("Enriching portfolio with live market data...")
    enriched = eng.enrich(port)

    eff_value = enriched.total_value

    # 2. Get or generate research brief
    brief_path = Path(brief) if brief else PROJECT_ROOT / "artifacts" / "research_brief.json"
    if not brief_path.exists():
        if not universe:
            print("Provide --universe or --brief to generate new plays.")
            print("Running portfolio analysis only (no new plays)...")
            rec = ActionRecommender()
            report = rec.recommend(enriched)
            print(format_portfolio_report(enriched, report))
            return

        logger.info(f"Running Market Research FSM on: {universe}")
        from research.fsm import MarketResearchFSM
        symbols = [s.strip() for s in universe.split(",")]
        fsm = MarketResearchFSM(
            project_root=PROJECT_ROOT,
            universe=symbols,
            dry_run=dry_run,
        )
        fsm.run()
        brief_path = PROJECT_ROOT / "artifacts" / "research_brief.json"

    # 3. Generate plays
    logger.info("Generating trade tickets...")
    gen = PlayGenerator(
        portfolio_value=eff_value,
        risk_pct=risk_pct,
        prefer_options=not no_options,
    )
    book = gen.generate(brief_path=brief_path)

    # 4. Recommend
    rec = ActionRecommender()
    report = rec.recommend(enriched, playbook=book)

    # 5. Print
    print(format_portfolio_report(enriched, report))

    if output:
        import json as _json
        Path(output).write_text(_json.dumps(report.to_dict(), indent=2))
        logger.info(f"Report saved to {output}")
    else:
        out = PROJECT_ROOT / "artifacts" / "recommendation.json"
        import json as _json
        out.write_text(_json.dumps(report.to_dict(), indent=2))
        logger.info(f"Report saved to {out}")


def main():
    cli()


if __name__ == "__main__":
    main()
