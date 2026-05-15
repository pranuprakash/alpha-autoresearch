# CLAUDE.md — Alpha Autoresearch Context

This file is for Claude (and other AI agents) to quickly understand the state of this project and continue development.

## What This Is

**Alpha Autoresearch** — Karpathy's [autoresearch](https://github.com/karpathy/autoresearch) applied to financial markets. Autonomous agents iterate on trading strategy code, backtest, keep or revert based on Sharpe ratio, and evolve through Darwinian selection. No GPU. No OpenClaw. Just API keys.

Inspired by [ATLAS by General Intelligence Capital](https://github.com/chrisworsey55/atlas-gic) which ran 378 trading days with this pattern and achieved +22% returns.

## Current State (as of 2026-03-17)

**Fully implemented and tested.** All 12 planned todos are complete. The system runs end-to-end.

### What Works

- `python prepare.py --universe "SPY"` — downloads data via yfinance, caches as parquet, splits train/val/test with embargo
- `python prepare.py --evaluate` — runs strategy.py through the full backtest harness, prints metrics
- `python main.py evaluate` — same via CLI
- `python main.py status` — shows experiment history
- `python main.py run --mode solo` — starts Karpathy-pure autonomous loop (needs API key in .env)
- `python main.py run --mode swarm` — starts multi-agent loop with Darwinian weights (needs API key in .env)
- All imports verified: backtest, risk, core.agent, core.tools, core.git_ops all work
- Integration tests pass: read_file, write_file (permission-guarded), run_backtest, results.tsv logging

### Verified Output (Baseline SPY Strategy)

```
train_sharpe:  0.1769
val_sharpe:   -0.2728    ← lots of room to improve
oos_ratio:    -1.54
```

The baseline is a simple momentum crossover (fast/slow MA). Agents have a lot to work with.

## Architecture

### The Karpathy Trinity

These three files are the core:

| File | Role | Modified by |
|------|------|-------------|
| `prepare.py` | Data download, splits, backtest harness | NEVER (read-only) |
| `strategy.py` | The trading strategy | ONLY the agent |
| `program.md` | Instructions for agents | The human |

### The Loop (Solo Mode)

```
1. Agent reads program.md + strategy.py + results.tsv
2. Agent modifies strategy.py via write_file tool
3. git commit
4. Backtest on train + validation
5. val_sharpe improved? KEEP (advance branch) : DISCARD (git reset)
6. Log to results.tsv
7. NEVER STOP
```

### The Swarm Loop

```
Layer 1 (Research):  macro + sentiment + technical agents (parallel)
Layer 2 (Strategy):  strategist synthesizes research -> modifies strategy.py
Layer 3 (Optimize):  optimizer runs Karpathy iterations (up to 5)
Layer 4 (Audit):     auditor checks for overfitting/bias/bugs
-> KEEP or REVERT -> update Darwinian weights -> repeat
```

### No OpenClaw, No litellm

The old `trading_research/` system used `openclaw agent` subprocess calls. This system uses the **`openai` SDK** for direct LLM API calls with provider routing via `core/agent.py:_make_client()`.

**NEVER add litellm** — it had a supply-chain compromise. Multi-provider routing is handled natively: `anthropic/*` routes to `api.anthropic.com/v1`, `google/*` to Google's OpenAI-compat endpoint, `fireworks/*` to Fireworks, `openai/*` to OpenAI directly.

## File Map

```
alpha_autoresearch/
├── prepare.py              ← Karpathy's prepare.py equivalent (DO NOT MODIFY)
├── strategy.py             ← Karpathy's train.py equivalent (AGENTS EDIT THIS)
├── program.md              ← Karpathy's program.md equivalent (HUMAN EDITS THIS)
├── config.yaml             ← Models, data config, backtest params
├── .env.example            ← API key template
│
├── core/
│   ├── agent.py            ← Agent class with function-calling tool loop (openai SDK, multi-provider routing)
│   ├── tools.py            ← 4 tools: read_file, write_file, run_backtest, list_files
│   ├── loop.py             ← Solo autoresearch loop
│   ├── swarm.py            ← Multi-agent orchestrator + Darwinian weights
│   └── git_ops.py          ← git commit/revert/branch/log_to_results
│
├── backtest/
│   ├── engine.py           ← VectorizedBacktester
│   ├── fees.py             ← FeeSchedule (equity/polymarket/kalshi)
│   ├── slippage.py         ← Square-root impact model
│   ├── splitter.py         ← Embargo-aware temporal splits (60/20/20)
│   └── metrics.py          ← Sharpe, Sortino, drawdown, Calmar, win rate
│
├── data/
│   ├── providers.py        ← yfinance downloader with parquet caching
│   └── cache/              ← Downloaded data (gitignored)
│
├── risk/
│   ├── guards.py           ← OOS ratio, plausibility, trade count checks
│   ├── audit.py            ← Look-ahead bias detection, overfitting signals
│   └── multiple_testing.py ← Bonferroni correction
│
├── strategies/templates/   ← momentum.py, mean_reversion.py, options_flow.py
├── prompts/                ← researcher.md, strategist.md, optimizer.md, auditor.md
├── artifacts/              ← Run logs, swarm_state.json
├── scripts/dashboard.py    ← TUI results viewer
└── main.py                 ← CLI entry point (click-based)
```

## Key Design Decisions

1. **Agents edit code, not configs** — they write the complete `strategy.py` each iteration
2. **Permission guard** — `ToolHandlers.WRITABLE_FILES = {"strategy.py"}` — checked against the *resolved* path, so `./strategy.py` and traversal variants are blocked. Agents cannot touch anything else.
3. **Git is the state machine** — branches track experiments, commit = keep, reset = discard
4. **The test set is sacred** — `--split test` in prepare.py is the one-shot holdout, never used during optimization
5. **Metrics are law** — `val_sharpe` decides keep/revert, not the agent's opinion
6. **OOS ratio guard** — `val_sharpe / train_sharpe >= 0.50` required to keep (prevents overfitting)

## Known Issues / Next Steps

### Immediate
- The default universe in `config.yaml` is 9 symbols (SPY, QQQ, AAPL, NVDA, etc.). When `--evaluate` runs without `--universe`, it downloads all 9 and picks the primary by volume. Consider defaulting to a single symbol for speed.
- `results.tsv` is cleaned up in the integration test but not auto-committed to git (intentional, matching Karpathy's design)

### What to Build Next
1. **Parallel research agents** — `swarm.py` currently runs researchers sequentially. Use `concurrent.futures.ThreadPoolExecutor` to run all 3 in parallel (3x faster)
2. **Equity curve plotting** — `scripts/dashboard.py` shows tabular results; add matplotlib equity curve
3. **Multi-symbol strategies** — `strategy.py` currently gets single-symbol data. Extend to cross-sectional (portfolio) strategies
4. **Options flow data** — `strategies/templates/options_flow.py` uses volume as proxy; wire in real CBOE/Unusual Whales data
5. **FRED macro data** — add macro indicators (VIX, yield curve, etc.) as optional features
6. **Parallel backtesting** — run multiple strategy variants in parallel instead of sequentially
7. **Better convergence detection** — currently checks "no keeps in last 10 experiments"; could be smarter

### Integration with existing `trading_research/`
The existing `trading_research/` directory has a 32-stage FSM pipeline that's OpenClaw-dependent. The two repos are complementary:
- `alpha_autoresearch/` = continuous optimization loop (pure Karpathy)
- `trading_research/` = one-time research pipeline (idea -> alpha thesis)

Future idea: `alpha_autoresearch/` generates the optimal strategy.py, then `trading_research/` handles the full 32-stage deployment pipeline (OMS, paper trading, risk audit).

## Running It

```bash
# Setup
pip install -e .
cp .env.example .env
# Add ANTHROPIC_API_KEY=sk-ant-... (or OPENAI_API_KEY= or GOOGLE_API_KEY=)

# Download data
python prepare.py --universe "SPY,QQQ" --period-start "2020-01-01" --period-end "2025-12-31"

# Verify the baseline
python main.py evaluate

# Launch solo mode (runs forever, Ctrl+C to stop)
python main.py run --mode solo --topic "SPY momentum optimization"

# Launch swarm mode
python main.py run --mode swarm --topic "multi-factor equity strategies"

# Check results
python main.py status
python scripts/dashboard.py
```

## Config Reference

```yaml
# config.yaml
models:
  solo_agent: "anthropic/claude-sonnet-4-20250514"   # single agent for solo mode
  researcher: "google/gemini-2.5-flash"               # research agents (cheaper)
  strategist: "anthropic/claude-sonnet-4-20250514"    # strategy modification
  optimizer: "anthropic/claude-sonnet-4-20250514"     # parameter optimization
  auditor: "openai/gpt-4o"                            # risk audit

backtest:
  initial_capital: 100000.0
  fee_model: "equity"          # equity | polymarket | kalshi
  time_budget_sec: 300
  train_pct: 0.60
  val_pct: 0.20
  test_pct: 0.20
  embargo_pct: 0.02            # gap between splits to prevent leakage

risk:
  min_oos_ratio: 0.50          # val_sharpe / train_sharpe minimum
  max_plausible_sharpe: 6.0    # anything higher is likely a bug
  min_trades: 30               # minimum for statistical significance
```

## Agent Contract

Every agent has access to these tools (defined in `core/tools.py`):

| Tool | Description | Restrictions |
|------|-------------|--------------|
| `read_file(path)` | Read any file in the project | Blocks `.env` and files matching `BLOCKED_READ_PATTERNS` |
| `write_file(path, content)` | Write a file | ONLY `strategy.py` (checked by resolved path, not raw string) |
| `run_backtest()` | Run `python prepare.py --evaluate` | Scans `strategy.py` for banned patterns first |
| `list_files(path)` | List directory contents | None |

**Security hardening in `core/tools.py`:**
- `_resolve()` uses `relative_to()` for path boundary checks (not `startswith()`) and rejects `..` segments
- `write_file` resolves the path before checking the allowlist — `./strategy.py` and traversal variants are blocked
- `run_backtest` rejects `strategy.py` containing `import os`, `subprocess`, `exec(`, `eval(`, `socket`, etc.
- `read_file` blocks `.env` reads

**`core/git_ops.py` safety:**
- `revert_last()` refuses to `git reset --hard` on `main`/`master` branches
- `revert_last()` only reverts commits whose message starts with `experiment:` (loop-made commits)

The `strategy.py` contract: must define `generate_signals(df: pd.DataFrame) -> pd.Series` returning positions (+1, -1, 0) with the same index as `df`.
