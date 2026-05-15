"""
Agent Tools — functions the LLM can call via function-calling.

Tools are pure Python functions. They're registered as OpenAI-format tool
schemas and executed locally when the LLM invokes them.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Tool schemas (OpenAI function-calling format)
# ---------------------------------------------------------------------------

TOOL_SCHEMAS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file. Use this to inspect strategy.py, results.tsv, program.md, or any other file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path from the project root (e.g. 'strategy.py', 'results.tsv')",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write the full contents of a file. Use this to modify strategy.py with your proposed changes. Always write the COMPLETE file, not a diff.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path from the project root (e.g. 'strategy.py')",
                    },
                    "content": {
                        "type": "string",
                        "description": "The complete new file content",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_backtest",
            "description": "Execute the backtest harness. Runs prepare.py --evaluate which backtests strategy.py on train and validation sets, returning Sharpe ratios and other metrics.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files in a directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative directory path (default: '.')",
                    },
                },
                "required": [],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Tool handler implementations
# ---------------------------------------------------------------------------

class ToolHandlers:
    """
    Concrete implementations of agent tools, bound to a project root.
    """

    # strategy.py is the ONLY file the agent should write to
    WRITABLE_FILES = {"strategy.py"}

    def __init__(self, project_root: Path):
        self.root = project_root.resolve()

    def _resolve(self, path: str) -> Path:
        if '\x00' in path or '..' in path.split('/'):
            raise PermissionError(f"Invalid path: {path}")
        resolved = (self.root / path).resolve()
        try:
            resolved.relative_to(self.root)
        except ValueError:
            raise PermissionError(f"Path escapes project root: {path}")
        return resolved

    # Files agents must never read (secrets, config with credentials)
    BLOCKED_READ_PATTERNS = {".env"}

    def read_file(self, path: str) -> str:
        if any(path == b or path.endswith('/' + b) for b in self.BLOCKED_READ_PATTERNS):
            return json.dumps({"error": f"Permission denied: {path}"})
        try:
            target = self._resolve(path)
        except PermissionError as e:
            return json.dumps({"error": str(e)})
        if not target.exists():
            return json.dumps({"error": f"File not found: {path}"})
        try:
            content = target.read_text()
            return content
        except Exception as e:
            return json.dumps({"error": str(e)})

    def write_file(self, path: str, content: str) -> str:
        try:
            target = self._resolve(path)
        except PermissionError as e:
            return json.dumps({"error": str(e)})
        allowed = {(self.root / f).resolve() for f in self.WRITABLE_FILES}
        if target not in allowed:
            return json.dumps({
                "error": f"Permission denied. Only these files are writable: {self.WRITABLE_FILES}. "
                         f"You tried to write to: {path}"
            })
        try:
            target.write_text(content)
            return json.dumps({"success": True, "path": path, "bytes": len(content)})
        except Exception as e:
            return json.dumps({"error": str(e)})

    # Patterns that should never appear in strategy.py (code execution risks)
    BANNED_STRATEGY_PATTERNS = [
        r'\bimport\s+os\b', r'\bimport\s+subprocess\b', r'\bimport\s+shutil\b',
        r'\b__import__\b', r'\beval\s*\(', r'\bexec\s*\(',
        r'\bos\.system\b', r'\bos\.popen\b', r'\bsocket\b',
    ]

    def run_backtest(self) -> str:
        """Run prepare.py --evaluate and capture output."""
        # Pre-execution safety scan on strategy.py
        strategy_path = self.root / "strategy.py"
        if strategy_path.exists():
            content = strategy_path.read_text()
            for pattern in self.BANNED_STRATEGY_PATTERNS:
                match = re.search(pattern, content)
                if match:
                    return json.dumps({
                        "error": f"strategy.py contains banned pattern: {match.group()!r}. "
                                 f"Remove dangerous imports/calls before running backtest."
                    })

        try:
            result = subprocess.run(
                [sys.executable, str(self.root / "prepare.py"), "--evaluate"],
                capture_output=True,
                text=True,
                timeout=600,
                cwd=str(self.root),
            )
            output = result.stdout
            if result.returncode != 0:
                output += f"\n\nSTDERR:\n{result.stderr}"
                output += f"\n\nExit code: {result.returncode}"
            return output
        except subprocess.TimeoutExpired:
            return json.dumps({"error": "Backtest exceeded 10-minute timeout"})
        except Exception as e:
            return json.dumps({"error": str(e)})

    def list_files(self, path: str = ".") -> str:
        target = self._resolve(path)
        if not target.is_dir():
            return json.dumps({"error": f"Not a directory: {path}"})
        try:
            entries = sorted(p.name for p in target.iterdir() if not p.name.startswith("."))
            return json.dumps(entries)
        except Exception as e:
            return json.dumps({"error": str(e)})

    def get_handlers(self) -> Dict[str, Any]:
        """Return a name -> callable mapping for Agent registration."""
        return {
            "read_file": self.read_file,
            "write_file": self.write_file,
            "run_backtest": self.run_backtest,
            "list_files": self.list_files,
        }
