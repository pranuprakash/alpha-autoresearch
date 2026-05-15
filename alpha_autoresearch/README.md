# Alpha Autoresearch

Karpathy's [autoresearch](https://github.com/karpathy/autoresearch) applied to financial markets. Agents iterate on trading strategy code, backtest, keep or revert based on Sharpe ratio, and evolve through Darwinian selection.

No GPU. No OpenClaw. Just API keys.

## Quick Start

```bash
# 1. Install
pip install -e .

# 2. Configure API keys
cp .env.example .env
# Edit .env with your LLM provider key(s)

# 3. Download data and prepare splits
python prepare.py --universe "SPY,QQQ,AAPL,NVDA"

# 4. Run a baseline backtest
python prepare.py --evaluate

# 5. Launch autonomous research (solo mode)
python -m alpha_autoresearch.main --mode solo

# 6. Launch swarm mode
python -m alpha_autoresearch.main --mode swarm --topic "momentum crossover optimization"
```

## How It Works

Three files that matter:

- **`prepare.py`** — data download, splits, backtest harness. Not modified by agents.
- **`strategy.py`** — the single file agents edit. Entry/exit logic, parameters, everything.
- **`program.md`** — instructions for the agents. Edited by you.

The loop:

1. Agent reads `program.md` + `strategy.py` + `results.tsv`
2. Agent modifies `strategy.py`
3. `git commit`
4. Backtest on train + validation sets
5. If `val_sharpe` improved: **keep**. If not: **git reset**.
6. Log to `results.tsv`
7. Repeat forever

## Modes

**Solo** — one agent, one file, one metric. Direct analogue to Karpathy's autoresearch.

**Swarm** — layered specialist agents (research, strategy, optimize, audit) with Darwinian weight evolution. Inspired by [ATLAS](https://github.com/chrisworsey55/atlas-gic).

## Cost

- Solo: ~$0.50–2.00 per experiment. 100 experiments overnight ≈ $50–200.
- Swarm: ~$2–5 per cycle. 50 cycles overnight ≈ $100–250.
- Infrastructure: any machine with Python.
