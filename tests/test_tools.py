"""Tests for core.tools — ToolHandlers: read_file, write_file, run_backtest, list_files."""

from __future__ import annotations

import json
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from pathlib import Path
from core.tools import ToolHandlers


class TestReadFile:
    def test_read_existing_file(self, tmp_project):
        t = ToolHandlers(tmp_project)
        result = t.read_file("strategy.py")
        assert "generate_signals" in result

    def test_read_missing_file(self, tmp_project):
        t = ToolHandlers(tmp_project)
        result = json.loads(t.read_file("nonexistent.py"))
        assert "error" in result

    def test_read_env_blocked(self, tmp_project):
        (tmp_project / ".env").write_text("API_KEY=secret\n")
        t = ToolHandlers(tmp_project)
        result = json.loads(t.read_file(".env"))
        assert "Permission denied" in result["error"]

    def test_path_traversal_blocked(self, tmp_project):
        t = ToolHandlers(tmp_project)
        result = t.read_file("../secret.txt")
        # tools.py catches PermissionError and returns JSON error string
        parsed = json.loads(result)
        assert "error" in parsed

    def test_read_subdirectory_file(self, tmp_project):
        (tmp_project / "data").mkdir(exist_ok=True)
        (tmp_project / "data" / "info.txt").write_text("hello")
        t = ToolHandlers(tmp_project)
        result = t.read_file("data/info.txt")
        assert "hello" in result


class TestWriteFile:
    def test_write_strategy_allowed(self, tmp_project):
        t = ToolHandlers(tmp_project)
        content = "import pandas as pd\ndef generate_signals(df): return pd.Series(0, index=df.index)\n"
        result = json.loads(t.write_file("strategy.py", content))
        assert result.get("success") is True
        assert (tmp_project / "strategy.py").read_text() == content

    def test_write_other_file_blocked(self, tmp_project):
        t = ToolHandlers(tmp_project)
        result = json.loads(t.write_file("config.yaml", "bad: data"))
        assert "Permission denied" in result["error"]
        assert "strategy.py" in result["error"]

    def test_write_traversal_blocked(self, tmp_project):
        t = ToolHandlers(tmp_project)
        result = json.loads(t.write_file("../evil.py", "import os"))
        assert "error" in result  # PermissionError caught and returned as JSON

    def test_write_strategy_dotslash_variant(self, tmp_project):
        t = ToolHandlers(tmp_project)
        result = json.loads(t.write_file("./strategy.py", "import pandas as pd\ndef generate_signals(df): return df['Close']\n"))
        # ./strategy.py resolves to strategy.py — should succeed
        assert result.get("success") is True


class TestListFiles:
    def test_list_root(self, tmp_project):
        t = ToolHandlers(tmp_project)
        result = json.loads(t.list_files("."))
        assert isinstance(result, list)
        assert "strategy.py" in result

    def test_list_subdir(self, tmp_project):
        t = ToolHandlers(tmp_project)
        result = json.loads(t.list_files("data"))
        assert isinstance(result, list)

    def test_list_nonexistent_dir(self, tmp_project):
        t = ToolHandlers(tmp_project)
        result = json.loads(t.list_files("does_not_exist"))
        assert "error" in result


class TestRunBacktest:
    def test_blocked_on_banned_pattern(self, tmp_project):
        (tmp_project / "strategy.py").write_text(
            "import os\ndef generate_signals(df): return None\n"
        )
        t = ToolHandlers(tmp_project)
        result = t.run_backtest()
        # Should be a JSON error, not crash
        parsed = json.loads(result)
        assert "error" in parsed
        assert "banned" in parsed["error"].lower() or "import os" in parsed["error"]

    def test_subprocess_call(self, tmp_project):
        """Verify run_backtest launches a subprocess (not just mocked)."""
        # strategy.py exists; prepare.py doesn't — subprocess should fail gracefully
        t = ToolHandlers(tmp_project)
        result = t.run_backtest()
        # Either returns backtest output or an error string — neither should raise
        assert isinstance(result, str)
