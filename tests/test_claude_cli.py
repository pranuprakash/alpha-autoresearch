"""
Tests for core.claude_cli — ClaudeCliAgent.

Most tests mock subprocess to avoid real API calls in CI.
One live smoke test is included (skip with -m 'not live').
"""

from __future__ import annotations

import json
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from unittest.mock import MagicMock, patch

import pytest
from core.claude_cli import (
    ClaudeCliAgent,
    _build_tool_system_section,
    _TOOL_CALL_RE,
)
from core.agent import AgentResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "File path"}},
                "required": ["path"],
            },
        },
    }
]

def _make_cli_json(result_text: str, session_id: str = "test-session") -> str:
    return json.dumps(
        {
            "type": "result",
            "subtype": "success",
            "is_error": False,
            "result": result_text,
            "session_id": session_id,
        }
    )


def _make_agent(handlers=None) -> ClaudeCliAgent:
    return ClaudeCliAgent(
        name="TestAgent",
        model="claude-cli/claude-sonnet-4-6",
        system_prompt="You are a test agent.",
        tools=SAMPLE_TOOL_SCHEMAS,
        tool_handlers=handlers or {},
    )


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

class TestToolSystemSection:
    def test_contains_tool_name(self):
        section = _build_tool_system_section(SAMPLE_TOOL_SCHEMAS)
        assert "read_file" in section

    def test_contains_protocol_header(self):
        section = _build_tool_system_section(SAMPLE_TOOL_SCHEMAS)
        assert "tool_call" in section.lower() or "TOOL USE" in section

    def test_empty_schemas_no_crash(self):
        section = _build_tool_system_section([])
        assert section == ""

    def test_required_args_flagged(self):
        section = _build_tool_system_section(SAMPLE_TOOL_SCHEMAS)
        assert "required" in section


class TestToolCallRegex:
    def test_simple_match(self):
        text = '<tool_call>{"name": "read_file", "arguments": {"path": "x.py"}}</tool_call>'
        matches = _TOOL_CALL_RE.findall(text)
        assert len(matches) == 1
        data = json.loads(matches[0])
        assert data["name"] == "read_file"

    def test_multiline_match(self):
        text = '<tool_call>\n{"name": "write_file", "arguments": {}}\n</tool_call>'
        matches = _TOOL_CALL_RE.findall(text)
        assert len(matches) == 1

    def test_no_match_on_plain_text(self):
        assert len(_TOOL_CALL_RE.findall("Hello, world!")) == 0


class TestClaudeCliAgentInit:
    def test_model_prefix_stripped(self):
        a = _make_agent()
        assert a.model == "claude-sonnet-4-6"

    def test_system_prompt_contains_tools(self):
        a = _make_agent()
        assert "read_file" in a._system_prompt

    def test_bare_model_no_prefix(self):
        a = ClaudeCliAgent(
            name="t", model="claude-sonnet-4-6",
            system_prompt="s", tools=[], tool_handlers={},
        )
        assert a.model == "claude-sonnet-4-6"


class TestClaudeCliAgentRun:
    def test_no_tool_calls_returns_content(self):
        a = _make_agent()
        with patch.object(a, "_call_cli", return_value=("Hello, world!", 0.5)):
            result = a.run("Say hello")
        assert result.content == "Hello, world!"
        assert result.tool_calls_made == 0
        assert result.error is None

    def test_one_tool_call_then_done(self):
        handler = MagicMock(return_value="file content here")
        a = _make_agent(handlers={"read_file": handler})

        responses = [
            ('<tool_call>{"name": "read_file", "arguments": {"path": "strategy.py"}}</tool_call>', 0.3),
            ("I read the file. It looks good.", 0.2),
        ]
        call_count = {"n": 0}

        def fake_call(msg, session_id, first_turn):
            idx = call_count["n"]
            call_count["n"] += 1
            return responses[idx]

        with patch.object(a, "_call_cli", side_effect=fake_call):
            result = a.run("Read the strategy file")

        assert result.tool_calls_made == 1
        assert "file content" in result.tool_results[0]["result_preview"]
        assert result.error is None

    def test_cli_error_returns_error_result(self):
        a = _make_agent()
        with patch.object(a, "_call_cli", side_effect=RuntimeError("CLI crashed")):
            result = a.run("any message")
        assert result.error is not None
        assert "CLI crashed" in result.error

    def test_unknown_tool_returns_error_in_result(self):
        a = _make_agent(handlers={})  # no handlers
        responses = [
            ('<tool_call>{"name": "nonexistent_tool", "arguments": {}}</tool_call>', 0.1),
            ("OK, I got an error.", 0.1),
        ]
        call_count = {"n": 0}

        def fake_call(msg, session_id, first_turn):
            idx = call_count["n"]
            call_count["n"] += 1
            return responses[idx]

        with patch.object(a, "_call_cli", side_effect=fake_call):
            result = a.run("use unknown tool")

        assert result.tool_calls_made == 1
        assert "Unknown tool" in result.tool_results[0]["result_preview"]

    def test_malformed_tool_call_json_recovers(self):
        a = _make_agent()
        responses = [
            ("<tool_call>NOT VALID JSON!!!</tool_call>", 0.1),
            ("OK, recovered.", 0.1),
        ]
        call_count = {"n": 0}

        def fake_call(msg, session_id, first_turn):
            idx = call_count["n"]
            call_count["n"] += 1
            return responses[idx]

        with patch.object(a, "_call_cli", side_effect=fake_call):
            result = a.run("trigger bad json")

        assert result.content == "OK, recovered."

    def test_max_rounds_exceeded(self):
        a = _make_agent(handlers={"read_file": lambda path: "data"})
        a.MAX_TOOL_ROUNDS = 3

        def always_tool(msg, session_id, first_turn):
            return ('<tool_call>{"name": "read_file", "arguments": {"path": "f"}}</tool_call>', 0.1)

        with patch.object(a, "_call_cli", side_effect=always_tool):
            result = a.run("keep calling")

        assert result.error is not None
        assert "MAX_TOOL_ROUNDS" in result.error

    def test_elapsed_time_tracked(self):
        a = _make_agent()
        with patch.object(a, "_call_cli", return_value=("done", 1.23)):
            result = a.run("hello")
        assert result.elapsed_sec == pytest.approx(1.23, abs=0.01)


class TestCallCli:
    def test_raises_on_nonzero_exit(self):
        a = _make_agent()
        mock_proc = MagicMock()
        mock_proc.returncode = 1
        mock_proc.stderr = "auth error"
        with patch("subprocess.run", return_value=mock_proc):
            with pytest.raises(RuntimeError, match="exited 1"):
                a._call_cli("test", "session-123", True)

    def test_raises_on_empty_output(self):
        a = _make_agent()
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = ""
        with patch("subprocess.run", return_value=mock_proc):
            with pytest.raises(RuntimeError, match="empty"):
                a._call_cli("test", "session-123", True)

    def test_raises_on_api_error_in_json(self):
        a = _make_agent()
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = json.dumps({"is_error": True, "result": "rate limited"})
        with patch("subprocess.run", return_value=mock_proc):
            with pytest.raises(RuntimeError, match="API error"):
                a._call_cli("test", "session-123", True)

    def test_extracts_result_field(self):
        a = _make_agent()
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = _make_cli_json("Expected output here")
        with patch("subprocess.run", return_value=mock_proc):
            text, elapsed = a._call_cli("test", "session-123", True)
        assert text == "Expected output here"

    def test_first_turn_uses_session_id(self):
        a = _make_agent()
        called_with = {}
        original_run = __import__("subprocess").run

        def capture_run(cmd, **kwargs):
            called_with["cmd"] = cmd
            mock_proc = MagicMock()
            mock_proc.returncode = 0
            mock_proc.stdout = _make_cli_json("ok")
            return mock_proc

        with patch("subprocess.run", side_effect=capture_run):
            a._call_cli("hello", "my-uuid-123", True)

        cmd = called_with["cmd"]
        assert "--session-id" in cmd
        assert "my-uuid-123" in cmd

    def test_resume_turn_uses_resume_flag(self):
        a = _make_agent()
        called_with = {}

        def capture_run(cmd, **kwargs):
            called_with["cmd"] = cmd
            mock_proc = MagicMock()
            mock_proc.returncode = 0
            mock_proc.stdout = _make_cli_json("ok")
            return mock_proc

        with patch("subprocess.run", side_effect=capture_run):
            a._call_cli("hello", "my-uuid-456", False)

        cmd = called_with["cmd"]
        assert "--resume" in cmd
        assert "my-uuid-456" in cmd


# ---------------------------------------------------------------------------
# Agent factory routing test
# ---------------------------------------------------------------------------

class TestAgentFactory:
    def test_claude_cli_prefix_returns_cli_agent(self):
        from core.agent import Agent
        a = Agent(
            name="test",
            model="claude-cli/claude-sonnet-4-6",
            system_prompt="Test",
        )
        assert isinstance(a, ClaudeCliAgent)

    def test_openai_prefix_returns_api_agent(self):
        from core.agent import Agent, _is_claude_cli_model
        assert not _is_claude_cli_model("openai/gpt-4o")

    def test_anthropic_prefix_not_cli(self):
        from core.agent import _is_claude_cli_model
        assert not _is_claude_cli_model("anthropic/claude-sonnet-4-20250514")

    def test_is_claude_cli_model_positive(self):
        from core.agent import _is_claude_cli_model
        assert _is_claude_cli_model("claude-cli/claude-sonnet-4-6")
        assert _is_claude_cli_model("claude-cli/claude-haiku-4-5")


# ---------------------------------------------------------------------------
# Live integration test (skipped unless --live flag or LIVE_TESTS env var)
# ---------------------------------------------------------------------------

@pytest.mark.live
def test_live_simple_response():
    """Live test: one call to the real Claude CLI. Requires claude installed + auth."""
    a = ClaudeCliAgent(
        name="LiveTest",
        model="claude-cli/claude-haiku-4-5",
        system_prompt="You are a helpful assistant.",
        tools=[],
        tool_handlers={},
    )
    result = a.run("Reply with exactly the word: PONG")
    assert result.error is None
    assert "PONG" in result.content.upper()
