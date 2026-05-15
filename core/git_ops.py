"""
Git operations for the autoresearch loop.

Branch = experiment state. Commit = keep. Reset = discard.
Mirrors Karpathy's approach: git is the only state machine.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Optional

logger = logging.getLogger("GitOps")


def _run(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, cwd=str(cwd))


def git_available(project_root: Path) -> bool:
    result = _run(["git", "status"], project_root)
    return result.returncode == 0


def current_branch(project_root: Path) -> str:
    result = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], project_root)
    return result.stdout.strip()


def current_commit(project_root: Path) -> str:
    result = _run(["git", "rev-parse", "--short", "HEAD"], project_root)
    return result.stdout.strip()


def has_changes(project_root: Path) -> bool:
    result = _run(["git", "status", "--porcelain"], project_root)
    return bool(result.stdout.strip())


def create_branch(project_root: Path, branch_name: str) -> bool:
    """Create and checkout a new branch for an experiment run."""
    existing = _run(["git", "branch", "--list", branch_name], project_root)
    if existing.stdout.strip():
        logger.info(f"Branch {branch_name} already exists, checking out")
        result = _run(["git", "checkout", branch_name], project_root)
    else:
        result = _run(["git", "checkout", "-b", branch_name], project_root)

    if result.returncode != 0:
        logger.error(f"Failed to create/checkout branch: {result.stderr}")
        return False

    logger.info(f"On branch: {branch_name}")
    return True


def commit_experiment(project_root: Path, description: str) -> Optional[str]:
    """Stage strategy.py and commit with experiment description."""
    _run(["git", "add", "strategy.py"], project_root)

    result = _run(
        ["git", "commit", "-m", f"experiment: {description}"],
        project_root,
    )

    if result.returncode != 0:
        if "nothing to commit" in result.stdout or "nothing to commit" in result.stderr:
            logger.warning("Nothing to commit — strategy.py unchanged")
            return None
        logger.error(f"Commit failed: {result.stderr}")
        return None

    commit_hash = current_commit(project_root)
    logger.info(f"Committed: {commit_hash} — {description}")
    return commit_hash


def revert_last(project_root: Path) -> bool:
    """Discard the last commit (git reset --hard HEAD~1).

    Safety guards:
    - Refuses on main/master branches
    - Only reverts commits made by the autoresearch loop (experiment: prefix)
    """
    branch = current_branch(project_root)
    if branch in ("main", "master"):
        logger.error(f"REFUSED: will not git reset on protected branch '{branch}'")
        return False

    # Only revert experiment commits made by the loop
    last_msg = _run(["git", "log", "-1", "--format=%s"], project_root)
    if not last_msg.stdout.strip().startswith("experiment:"):
        logger.error(
            f"REFUSED: last commit is not an experiment commit: "
            f"{last_msg.stdout.strip()!r}"
        )
        return False

    result = _run(["git", "reset", "--hard", "HEAD~1"], project_root)
    if result.returncode != 0:
        logger.error(f"Reset failed: {result.stderr}")
        return False
    logger.info(f"Reverted to {current_commit(project_root)}")
    return True


def log_to_results(
    project_root: Path,
    commit_hash: str,
    val_sharpe: float,
    train_sharpe: float,
    max_drawdown: float,
    status: str,
    description: str,
):
    """Append a row to results.tsv (untracked by git, like Karpathy's)."""
    results_path = project_root / "results.tsv"

    if not results_path.exists():
        results_path.write_text(
            "commit\tval_sharpe\ttrain_sharpe\tmax_drawdown\tstatus\tdescription\n"
        )

    row = (
        f"{commit_hash}\t{val_sharpe:.6f}\t{train_sharpe:.6f}\t"
        f"{max_drawdown:.6f}\t{status}\t{description}\n"
    )
    with open(results_path, "a") as f:
        f.write(row)

    logger.info(f"Logged: {status} | val_sharpe={val_sharpe:.4f} | {description}")


def init_results_tsv(project_root: Path):
    """Create results.tsv with header if it doesn't exist."""
    results_path = project_root / "results.tsv"
    if not results_path.exists():
        results_path.write_text(
            "commit\tval_sharpe\ttrain_sharpe\tmax_drawdown\tstatus\tdescription\n"
        )
        logger.info("Initialized results.tsv")
