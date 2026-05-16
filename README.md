# Alpha Autoresearch

> Karpathy's [autoresearch](https://github.com/karpathy/autoresearch) applied to financial markets. Agents iterate on trading strategy code, run vectorized backtests, keep or revert on Sharpe ratio, and evolve through Darwinian selection — all without human intervention.

**Three systems, one pipeline:**

```
Research FSM ──► Karpathy Loop ──► Play Generator + Portfolio Engine
 (alpha signals)   (strategy code)    (trade tickets + risk management)
```

No GPU required. Runs on a laptop. Uses Claude Max (no API bill) or any OpenAI-compatible endpoint.

---

## What It Does

**System 1 — Karpathy Loop** (`run` command)

A single agent or multi-agent swarm iterates on `strategy.py` forever. Each iteration: read, propose, backtest, git commit — keep if `val_sharpe` improved, `git reset` if not. The loop never stops until you tell it to.

- **Solo mode**: one agent, one file, direct analogue to Karpathy's original
- **Swarm mode**: layered specialists (researcher → strategist → optimizer → auditor) with Darwinian weight evolution — better-performing agents get louder over time

**System 2 — Market Research FSM** (`research` command)

A 12-stage finite state machine that autonomously discovers alpha signals before the loop runs. Quantitative stages (indicator scan, regime detection) feed into LLM synthesis stages (signal generation, strategy codegen, backtest validation). Resumable. Outputs `artifacts/research_brief.json`.

```
UNIVERSE_SCAN → MACRO_REGIME → TECHNICAL_SCAN → SENTIMENT → OPTIONS_FLOW
      → SIGNAL_SYNTHESIS → STRATEGY_CODEGEN → BACKTEST_VALIDATION → ALPHA_REPORT
```

**System 3 — Play Generator + Portfolio Engine** (`plays`, `portfolio`, `recommend`)

Converts research briefs into specific trade tickets. Fetches live options chains, computes Black-Scholes Greeks, sizes positions using Kelly criterion, and reconciles new plays against your existing portfolio constraints.

---

## Quick Start

```bash
# 1. Install
pip install -e .

# 2. Option A: Claude Max (no API bill)
#    Requires Claude Code installed: https://claude.ai/code
#    Set model prefix to "claude-cli/" in config.yaml (default)

# Option B: API keys
cp .env.example .env
# Edit .env with your provider key(s) and update config.yaml model prefix

# 3. Download market data
python main.py prepare --universe "SPY,QQQ,NVDA,META"

# 4. Verify the baseline backtest runs
python main.py evaluate

# 5. Start the autonomous loop (solo mode — runs forever)
python main.py run --mode solo --topic "momentum optimization"

# 6. Or run swarm mode with multi-agent Darwinian evolution
python main.py run --mode swarm

# 7. Research first, then optimize (recommended)
python main.py research --universe "NVDA,META,SPY" --dry-run   # fast, no LLM
python main.py run --mode solo --with-research --universe "NVDA,META,SPY"
```

---

## Model Configuration

Edit `config.yaml` to switch providers. The model prefix determines routing — no code changes needed:

```yaml
models:
  # Claude Max (local CLI, no API key required)
  solo_agent: "claude-cli/claude-sonnet-4-6"
  strategist: "claude-cli/claude-opus-4-7"

  # Anthropic API (ANTHROPIC_API_KEY in .env)
  # solo_agent: "anthropic/claude-sonnet-4-6"

  # OpenAI API (OPENAI_API_KEY in .env)
  # solo_agent: "openai/gpt-4o"

  # Google AI API (GOOGLE_API_KEY in .env)
  # solo_agent: "google/gemini-2.5-flash"
```

---

## The Karpathy Trinity

Three files. This is the entire interface.

| File | Role | Modified by |
|------|------|-------------|
| `prepare.py` | Data download, splits, backtest harness | **Never** — it's the referee |
| `strategy.py` | The trading strategy to be evolved | **Only the agent** |
| `program.md` | Instructions for the agent | **You** |

`strategy.py` must always define:

```python
def generate_signals(df: pd.DataFrame) -> pd.Series:
    """
    Input:  OHLCV DataFrame
    Output: Series of {-1, 0, +1} — short, flat, long
    """
```

---

## All CLI Commands

```bash
# ── System 1: Karpathy Loop ───────────────────────────────────────
python main.py run --mode solo                                # infinite solo loop
python main.py run --mode solo --topic "RSI mean-reversion"  # with focus topic
python main.py run --mode solo --max-iterations 20           # bounded run
python main.py run --mode swarm                              # multi-agent swarm
python main.py run --mode swarm --max-cycles 5               # bounded swarm
python main.py run --mode solo --with-research --universe "NVDA,META"

python main.py evaluate                                      # single backtest
python main.py status                                        # experiment history + swarm weights
python main.py test-run                                      # one-shot holdout test (SACRED)

# ── System 2: Market Research FSM ────────────────────────────────
python main.py research --universe "NVDA,META,SPY"           # full LLM run
python main.py research --universe "NVDA,META,SPY" --dry-run # quant-only, no LLM cost
python main.py research --resume <run_id>                    # resume an interrupted run
python main.py research --topic "AI infrastructure plays"

# ── System 3: Plays + Portfolio ──────────────────────────────────
python main.py plays --universe "NVDA,META"                  # generate trade tickets
python main.py plays --no-options                            # equity plays only
python main.py plays --portfolio-value 100000 --risk-pct 0.02

python main.py portfolio --file my_portfolio.json            # analyze holdings
python main.py recommend --file portfolio.json --universe "NVDA,META"  # full pipeline
python main.py recommend --file portfolio.json --dry-run     # no LLM cost
```

---

## Portfolio Input Format

```json
{
  "cash": 10000,
  "positions": [
    {
      "asset_type": "equity",
      "symbol": "NVDA",
      "shares": 10,
      "cost_basis": 850.0
    },
    {
      "asset_type": "option",
      "symbol": "META",
      "option_type": "call",
      "strike": 700,
      "expiry": "2026-10-16",
      "quantity": 1,
      "cost_basis": 31.75
    }
  ]
}
```

The portfolio engine fetches live prices, computes B-S Greeks for every option, aggregates net delta/theta/vega, estimates 1-day 95% VaR, and generates hold/close/trim/roll recommendations.

---

## Architecture

```
alpha_autoresearch/
├── prepare.py                     ← Data download + backtest harness (read-only)
├── strategy.py                    ← The strategy being evolved (agent-only)
├── program.md                     ← Agent instructions (human-editable)
├── config.yaml                    ← Models, data universe, backtest params
│
├── core/
│   ├── agent.py                   ← Multi-provider LLM router (claude-cli/openai/anthropic/google)
│   ├── claude_cli.py              ← Claude CLI subprocess backend (no API key)
│   ├── loop.py                    ← SoloLoop: Karpathy keep/revert logic
│   ├── swarm.py                   ← SwarmOrchestrator: Darwinian multi-agent
│   ├── git_ops.py                 ← git commit/reset state machine
│   └── tools.py                   ← 4 agent tools: read/write file, run backtest, list files
│
├── backtest/
│   ├── engine.py                  ← VectorizedBacktester (vectorized, no loops)
│   ├── splitter.py                ← Temporal train/val/test splits with embargo
│   ├── fees.py                    ← Fee schedules (equity, polymarket, kalshi)
│   └── metrics.py                 ← Sharpe, Sortino, max drawdown, win rate
│
├── research/                      ← System 2: Market Research FSM
│   ├── fsm.py                     ← MarketResearchFSM (12 states, resumable)
│   ├── states.py                  ← ResearchState enum + FsmContext
│   ├── indicators.py              ← 12+ technical indicators (pure numpy/pandas)
│   ├── agents.py                  ← Per-state ClaudeCliAgent builders
│   └── report.py                  ← Terminal-formatted alpha report
│
├── plays/                         ← System 3A: Play Generator
│   ├── generator.py               ← PlayGenerator: research brief → trade tickets
│   ├── models.py                  ← OptionPlay, EquityPlay, PlayBook dataclasses
│   ├── options.py                 ← Black-Scholes Greeks, options chain fetcher
│   └── sizing.py                  ← Kelly criterion, fixed-fraction, contract math
│
├── portfolio/                     ← System 3B: Portfolio Engine
│   ├── engine.py                  ← PortfolioEngine: load holdings + live enrichment
│   ├── models.py                  ← Position, Portfolio, PortfolioSummary
│   ├── risk.py                    ← Greeks aggregation, 1-day VaR
│   ├── recommender.py             ← ActionRecommender: hold/close/trim/roll
│   └── reporter.py                ← Terminal-formatted portfolio report
│
├── risk/
│   ├── guards.py                  ← OOS ratio, Sharpe plausibility, min trades
│   ├── audit.py                   ← Look-ahead bias detection (static analysis)
│   └── multiple_testing.py        ← Bonferroni correction
│
├── prompts/                       ← Swarm agent system prompts
│   ├── researcher.md              ← Macro/technical/sentiment analysis
│   ├── strategist.md              ← Strategy design with code examples
│   ├── optimizer.md               ← Parameter tuning heuristics
│   └── auditor.md                 ← Risk audit checklist + verdict format
│
└── tests/                         ← 289 tests, 1 skipped
    ├── test_research_fsm.py       ← 71 FSM tests
    ├── test_plays/                ← 35 play generator tests
    ├── test_portfolio/            ← 60 portfolio engine tests
    └── ...
```

---

## Design Principles

**No framework lock-in.** No litellm, no langchain. A clean `Agent` class with a routing table. Adding a new provider is ~10 lines.

**Vectorized backtesting.** The backtest engine runs on full-array pandas operations — no Python loops over bars. A 6-year daily backtest on SPY completes in ~50ms.

**No look-ahead bias by design.** Every agent prompt includes explicit examples showing correct `.shift(1)` usage. The auditor agent checks for violations.

**Git as the state machine.** Every experiment is a commit. Keep → git commit stays. Discard → git reset --hard HEAD~1. The full history is in the log.

**The holdout set is sacred.** `test-run` can only be called once per research session (it asks for confirmation). The test set is never used for strategy selection — only final reporting.

---

## Risk Guards

The loop enforces these automatically before keeping any experiment:

| Guard | Threshold | Rationale |
|-------|-----------|-----------|
| OOS ratio (`val_sharpe / train_sharpe`) | ≥ 0.50 | Detect overfitting |
| Max plausible Sharpe | < 6.0 | Flag look-ahead bias |
| Minimum trades | ≥ 30 | Statistical significance |
| Strategy code | No `os`, `subprocess`, `eval`, `exec` | Safety |
| `write_file` tool | Only `strategy.py` | Prevent agent drift |
| `git reset` | Blocked on `main`/`master` | Prevent destructive resets |

---

## Tests

```bash
python -m pytest                                          # 289 tests
python -m pytest tests/test_research_fsm.py              # FSM (71 tests)
python -m pytest tests/test_plays/ tests/test_portfolio/ # Systems 3A+3B (95 tests)
python -m pytest tests/test_claude_cli.py -k "not live"  # CLI tests (no real API call)
```

---

## License

MIT
