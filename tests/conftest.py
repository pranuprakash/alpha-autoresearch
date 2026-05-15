"""Shared fixtures for alpha_autoresearch tests."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Generator

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """Synthetic 500-row OHLCV DataFrame with a clear trend."""
    rng = np.random.default_rng(42)
    n = 500
    close = 100.0 + np.cumsum(rng.normal(0.1, 1.0, n))
    high = close + rng.uniform(0.5, 1.5, n)
    low = close - rng.uniform(0.5, 1.5, n)
    open_ = close + rng.normal(0, 0.5, n)
    volume = rng.integers(1_000_000, 10_000_000, n).astype(float)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=idx,
    )


@pytest.fixture
def tmp_project(tmp_path: Path) -> Generator[Path, None, None]:
    """
    Minimal project directory with strategy.py, prepare.py stub,
    config.yaml, and a fresh git repo.
    """
    (tmp_path / "artifacts").mkdir()
    (tmp_path / "data" / "cache").mkdir(parents=True)

    # strategy.py — minimal valid strategy
    (tmp_path / "strategy.py").write_text(
        "import pandas as pd\n"
        "def generate_signals(df):\n"
        "    return pd.Series(1, index=df.index)\n"
    )

    # program.md
    (tmp_path / "program.md").write_text("You are a trading researcher.\n")

    # config.yaml
    (tmp_path / "config.yaml").write_text(
        "models:\n"
        "  solo_agent: 'claude-cli/claude-sonnet-4-6'\n"
        "backtest:\n"
        "  initial_capital: 100000.0\n"
        "  fee_model: equity\n"
        "  time_budget_sec: 300\n"
        "  train_pct: 0.60\n"
        "  val_pct: 0.20\n"
        "  test_pct: 0.20\n"
        "  embargo_pct: 0.02\n"
        "risk:\n"
        "  min_oos_ratio: 0.50\n"
        "  max_plausible_sharpe: 6.0\n"
        "  min_trades: 30\n"
        "loop:\n"
        "  max_iterations: null\n"
        "  convergence_threshold: 0.01\n"
    )

    # init git
    import subprocess
    subprocess.run(["git", "init"], cwd=str(tmp_path), capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=str(tmp_path), capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=str(tmp_path), capture_output=True,
    )
    subprocess.run(
        ["git", "add", "strategy.py", "program.md", "config.yaml"],
        cwd=str(tmp_path), capture_output=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "initial"],
        cwd=str(tmp_path), capture_output=True,
    )

    yield tmp_path
