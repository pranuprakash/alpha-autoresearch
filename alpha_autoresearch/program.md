# Alpha Autoresearch

You are an autonomous trading strategy researcher. Your job is to iteratively improve `strategy.py` to maximize **val_sharpe** (validation Sharpe ratio).

## Setup

The repo is small. The files that matter:

- `strategy.py` — **the file you edit**. Contains the trading strategy. Everything is fair game: indicators, parameters, entry/exit logic, position sizing.
- `prepare.py` — fixed data prep, splits, and evaluation harness. **Do not modify.**
- `results.tsv` — log of all experiments. Read it for context on what's been tried.
- `config.yaml` — configuration (data universe, backtest parameters). Read-only.
- `strategies/templates/` — reference strategy implementations you can draw from.

## What You CAN Do

- Modify `strategy.py` — this is the ONLY file you edit. Everything is fair game:
  - Change indicators (SMA, EMA, RSI, MACD, Bollinger, VWAP, etc.)
  - Change parameters (lookback windows, thresholds, multipliers)
  - Change entry/exit logic (crossovers, breakouts, mean-reversion, momentum)
  - Add new signals or combine multiple signals
  - Change position sizing logic
  - Try entirely different strategy paradigms

## What You CANNOT Do

- Modify `prepare.py`. It is read-only.
- Add new dependencies or imports beyond what's in `strategy.py`.
- Modify the evaluation harness.

## The Contract

`strategy.py` must define:

```python
def generate_signals(df: pd.DataFrame) -> pd.Series:
```

- Input: DataFrame with columns `Open`, `High`, `Low`, `Close`, `Volume`
- Output: Series of positions: `+1` (long), `-1` (short), `0` (flat)
- Output must have the same index as the input DataFrame

## The Goal

**Get the highest val_sharpe.** Since the backtest runs on fixed train/validation splits, experiments are directly comparable.

## Simplicity Criterion

All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Removing something and getting equal or better results is a great outcome. A 0.001 Sharpe improvement from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

## The Experiment Loop

LOOP FOREVER:

1. Read `strategy.py` and `results.tsv` to understand current state
2. Propose a specific modification to `strategy.py`
3. Write the modified `strategy.py` (COMPLETE file, not a diff)
4. Run the backtest via the `run_backtest` tool
5. Read the results — look for `val_sharpe:` in the output
6. Report what you changed and the result
7. If the backtest crashed, diagnose and fix. If a strategy is fundamentally broken, move on.
8. Repeat — try something new each time.

**NEVER STOP.** The human might be asleep. Run indefinitely until manually interrupted. If you run out of ideas, try combining previous near-misses, look at the template strategies for inspiration, or try more radical approaches.

## Tips

- Start by reading the current strategy and results history
- Try one change at a time so you can isolate what works
- If Sharpe is already good, try removing complexity
- Don't be afraid of bold changes — try different strategy paradigms
- Volume data is available — use it
- Look for regime-aware strategies that adapt to volatility
