"""Tests for core.git_ops — commit, revert, branch, results logging."""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pathlib import Path
import pytest
from core.git_ops import (
    commit_experiment,
    create_branch,
    current_branch,
    current_commit,
    git_available,
    init_results_tsv,
    log_to_results,
    revert_last,
)


class TestGitAvailable:
    def test_available_in_git_repo(self, tmp_project):
        assert git_available(tmp_project) is True

    def test_unavailable_in_non_git_dir(self, tmp_path):
        non_git = tmp_path / "not_a_repo"
        non_git.mkdir()
        assert git_available(non_git) is False


class TestCurrentBranch:
    def test_returns_branch_name(self, tmp_project):
        branch = current_branch(tmp_project)
        assert isinstance(branch, str)
        assert len(branch) > 0


class TestCurrentCommit:
    def test_returns_hash(self, tmp_project):
        h = current_commit(tmp_project)
        assert isinstance(h, str)
        assert len(h) >= 7


class TestCreateBranch:
    def test_creates_new_branch(self, tmp_project):
        ok = create_branch(tmp_project, "autoresearch/test-run")
        assert ok is True
        assert current_branch(tmp_project) == "autoresearch/test-run"

    def test_checkout_existing_branch(self, tmp_project):
        create_branch(tmp_project, "autoresearch/reuse")
        import subprocess
        subprocess.run(["git", "checkout", "main"], cwd=str(tmp_project), capture_output=True)
        ok = create_branch(tmp_project, "autoresearch/reuse")
        assert ok is True


class TestCommitExperiment:
    def test_commit_strategy_change(self, tmp_project):
        (tmp_project / "strategy.py").write_text(
            "import pandas as pd\n"
            "def generate_signals(df):\n"
            "    return pd.Series(-1, index=df.index)\n"
        )
        hash_ = commit_experiment(tmp_project, "test change")
        assert hash_ is not None
        assert len(hash_) >= 7

    def test_no_commit_on_unchanged_file(self, tmp_project):
        # File unchanged since last commit
        result = commit_experiment(tmp_project, "unchanged")
        assert result is None  # "nothing to commit"


class TestRevertLast:
    def test_refuses_on_main_branch(self, tmp_project):
        branch = current_branch(tmp_project)
        assert branch in ("main", "master"), f"Expected main/master, got {branch!r}"
        ok = revert_last(tmp_project)
        assert ok is False

    def test_reverts_experiment_commit(self, tmp_project):
        create_branch(tmp_project, "autoresearch/revert-test")
        original = (tmp_project / "strategy.py").read_text()

        (tmp_project / "strategy.py").write_text(
            "import pandas as pd\n"
            "def generate_signals(df): return pd.Series(0, index=df.index)\n"
        )
        commit_experiment(tmp_project, "modify for revert test")
        assert (tmp_project / "strategy.py").read_text() != original

        ok = revert_last(tmp_project)
        assert ok is True
        assert (tmp_project / "strategy.py").read_text() == original

    def test_refuses_non_experiment_commit(self, tmp_project):
        import subprocess
        create_branch(tmp_project, "autoresearch/non-exp")
        (tmp_project / "strategy.py").write_text("# changed\nimport pandas as pd\ndef generate_signals(df): return df['Close']\n")
        subprocess.run(
            ["git", "add", "strategy.py"],
            cwd=str(tmp_project), capture_output=True,
        )
        subprocess.run(
            ["git", "commit", "-m", "manual: not an experiment"],
            cwd=str(tmp_project), capture_output=True,
        )
        ok = revert_last(tmp_project)
        assert ok is False  # refused — not "experiment:" prefix


class TestResultsLog:
    def test_init_creates_header(self, tmp_project):
        init_results_tsv(tmp_project)
        path = tmp_project / "results.tsv"
        assert path.exists()
        lines = path.read_text().splitlines()
        assert lines[0].startswith("commit\t")

    def test_log_appends_row(self, tmp_project):
        init_results_tsv(tmp_project)
        log_to_results(tmp_project, "abc1234", 0.75, 1.0, -0.12, "keep", "test experiment")
        lines = (tmp_project / "results.tsv").read_text().splitlines()
        assert len(lines) == 2
        assert "abc1234" in lines[1]
        assert "keep" in lines[1]

    def test_multiple_appends(self, tmp_project):
        init_results_tsv(tmp_project)
        for i in range(5):
            log_to_results(tmp_project, f"commit{i}", float(i) * 0.1, 0.5, -0.05, "keep", f"exp{i}")
        lines = (tmp_project / "results.tsv").read_text().splitlines()
        assert len(lines) == 6  # header + 5 data rows
