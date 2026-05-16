# CLAUDE.md — Project Context for AI Agents

This file helps Claude (and other AI coding agents) understand the state of this project quickly.

## What This Is

**Alpha Autoresearch** — Karpathy-style autonomous trading research. Agents iterate on `strategy.py`, backtest against Sharpe ratio, keep or revert. Extended with a full market research FSM, play generator, and portfolio engine for live trading decisions.

## Current State (as of 2026-05-15)

**Three fully-shipped systems, 289 tests passing.**

### System 1: Karpathy Loop (`run` command)
- Solo mode: one agent iterates strategy.py, git keeps/reverts on val_sharpe
- Swarm mode: research → strategist → optimizer → auditor with Darwinian weights
- Baseline SPY: train_sharpe=0.1769, val_sharpe=-0.2728 (room to improve)
- Uses `openai` SDK for direct LLM calls. **NEVER add litellm** (had a supply-chain compromise; not used here)

### System 2: Market Research FSM (`research` command)
12-stage autonomous pipeline: IDLE → UNIVERSE_SCAN → MACRO_REGIME → TECHNICAL_SCAN → SENTIMENT → OPTIONS_FLOW → SIGNAL_SYNTHESIS → STRATEGY_CODEGEN → BACKTEST_VALIDATION → ALPHA_REPORT → COMPLETE

- Quantitative stages (UNIVERSE_SCAN, MACRO_REGIME, TECHNICAL_SCAN): pure math, no LLM
- LLM stages (SIGNAL_SYNTHESIS, STRATEGY_CODEGEN): call Claude via `claude-cli/` model prefix
- `--dry-run`: skips LLM stages, uses quant signals only (fast, no API cost)
- Output: `artifacts/research_brief.json` — feeds into System 1 and System 3
- State persisted after each transition; resumable with `--resume <run_id>`

### System 3: Play Generator + Portfolio Engine (`plays`, `portfolio`, `recommend`)
- **Play Generator** (`plays/`): reads research_brief.json → exact trade tickets
  - Fetches live options chain via yfinance; selects strike at target delta (0.40 default)
  - Black-Scholes Greeks; IV rank (52-week realized vol proxy); Kelly/fixed-fraction sizing
  - Falls back to equity play if no liquid chain
- **Portfolio Engine** (`portfolio/`): loads your holdings JSON → live enrichment → recommendations
  - Computes live P&L, B-S Greeks per option, net portfolio delta/theta/vega, 1-day 95% VaR
  - `ActionRecommender`: HOLD/CLOSE/TRIM/ROLL per position; filters new plays by concentration + cash + risk budget

## Karpathy Trinity (NEVER confuse)

| File | Role | Modified by |
|------|------|-------------|
| `prepare.py` | Data download + backtest harness | NEVER (read-only) |
| `strategy.py` | The trading strategy | ONLY the agent |
| `program.md` | Agent instructions | The human |

## File Map

```
alpha_autoresearch/
├── prepare.py, strategy.py, program.md   ← Karpathy Trinity
├── config.yaml                           ← models, data, backtest params
│
├── core/                                 ← Karpathy loop internals
│   ├── agent.py                          ← LLM tool loop (openai SDK)
│   ├── loop.py / swarm.py               ← solo + swarm orchestrators
│   └── git_ops.py, tools.py             ← git state machine + 4 tools
│
├── backtest/                             ← VectorizedBacktester, fees, metrics
├── data/providers.py                     ← yfinance downloader + parquet cache
├── risk/                                 ← OOS ratio, look-ahead bias, Bonferroni
│
├── research/                             ← Market Research FSM (System 2)
│   ├── fsm.py                            ← MarketResearchFSM main engine
│   ├── states.py                         ← ResearchState enum + FsmContext
│   ├── indicators.py                     ← 12+ quant indicators (pure math)
│   ├── agents.py                         ← per-state ClaudeCliAgent builders
│   └── report.py                         ← alpha report formatter
│
├── plays/                                ← Play Generator (System 3A)
│   ├── generator.py                      ← PlayGenerator: brief → trade tickets
│   ├── models.py                         ← OptionPlay, EquityPlay, PlayBook
│   ├── options.py                        ← B-S Greeks, chain fetch, IV rank
│   └── sizing.py                         ← Kelly, fixed-frac, contract math
│
├── portfolio/                            ← Portfolio Engine (System 3B)
│   ├── engine.py                         ← PortfolioEngine: load + enrich
│   ├── models.py                         ← Position, Portfolio dataclasses
│   ├── risk.py                           ← Greeks aggregation, VaR
│   ├── recommender.py                    ← ActionRecommender
│   └── reporter.py                       ← terminal-formatted output
│
├── artifacts/                            ← research_brief.json, playbook.json, logs
└── main.py                               ← CLI entry (click)
```

## All CLI Commands

```bash
# System 1 — Karpathy Loop
python main.py run --mode solo --topic "..."          # solo forever
python main.py run --mode swarm                       # multi-agent Darwinian
python main.py run --mode solo --with-research --universe "NVDA,META"  # FSM → loop
python main.py evaluate                               # single backtest
python main.py status                                 # experiment history

# System 2 — Market Research FSM
python main.py research --universe "NVDA,META,SPY" --dry-run  # quant-only, fast
python main.py research --universe "NVDA,META,SPY"             # full LLM run
python main.py research --resume <run_id>                      # resume interrupted run

# System 3 — Plays + Portfolio
python main.py plays --universe "NVDA,META" --portfolio-value 100000
python main.py portfolio --file my_portfolio.json
python main.py recommend --file portfolio.json --universe "NVDA,META" [--dry-run]
```

## Portfolio JSON Format

```json
{
  "cash": 10000,
  "positions": [
    {"asset_type": "equity", "symbol": "NVDA", "shares": 5, "cost_basis": 900.0},
    {"asset_type": "option", "symbol": "META", "option_type": "call",
     "strike": 700, "expiry": "2026-10-16", "quantity": 1, "cost_basis": 31.75},
    {"asset_type": "bond", "symbol": "TLT", "shares": 20, "cost_basis": 88.0}
  ]
}
```

## Risk Guards (Karpathy loop)

- `val_sharpe / train_sharpe >= 0.50` (OOS ratio)
- `Sharpe < 6.0` (plausibility cap)
- `trades >= 30` (statistical minimum)
- `strategy.py` cannot contain `import os`, `subprocess`, `exec(`, `eval(`, `socket`
- `write_file` allowlist: ONLY `strategy.py` (path-resolved, traversal-blocked)
- `git reset --hard` blocked on `main`/`master`; only reverts `experiment:` commits

## Tests

```bash
/path/to/venv/bin/python -m pytest           # 289 passed, 1 skipped (live CLI)
/path/to/venv/bin/python -m pytest tests/test_plays/ tests/test_portfolio/  # 95 tests
/path/to/venv/bin/python -m pytest tests/test_research_fsm.py               # 71 tests
```
