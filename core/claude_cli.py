"""
ClaudeCliAgent — agent backend using the `claude` CLI subprocess.

No API keys. Uses Pranu's Max subscription via the local `claude` CLI.
Implements pseudo-function-calling via XML-tagged JSON blocks so the
same tool loop works as with the OpenAI SDK backend.

Multi-turn conversation is maintained via claude's --session-id / --resume flags.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("ClaudeCli")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CLAUDE_CLI = shutil.which("claude") or "/Users/pranuprakash/.local/bin/claude"


def verify_claude_cli() -> None:
    """Raise RuntimeError early if the claude CLI is missing or broken."""
    import os
    if not os.path.isfile(CLAUDE_CLI):
        raise RuntimeError(
            f"claude CLI not found at {CLAUDE_CLI}. "
            "Install Claude Code: https://claude.ai/code"
        )
    try:
        proc = subprocess.run(
            [CLAUDE_CLI, "--version"],
            capture_output=True, text=True, timeout=10
        )
        if proc.returncode != 0:
            raise RuntimeError(f"claude CLI --version failed: {proc.stderr.strip()[:200]}")
    except FileNotFoundError:
        raise RuntimeError(f"claude CLI not executable at {CLAUDE_CLI}")
    except subprocess.TimeoutExpired:
        raise RuntimeError("claude CLI --version timed out")

# XML tags used to delimit tool calls and results in the prompt
TOOL_CALL_TAG = "tool_call"
TOOL_RESULT_TAG = "tool_result"

# Pattern to extract <tool_call>...</tool_call> blocks
_TOOL_CALL_RE = re.compile(
    rf"<{TOOL_CALL_TAG}>\s*(.*?)\s*</{TOOL_CALL_TAG}>",
    re.DOTALL,
)


def _build_tool_system_section(tool_schemas: List[Dict[str, Any]]) -> str:
    """
    Format OpenAI-style tool schemas as readable tool descriptions
    for the system prompt.
    """
    if not tool_schemas:
        return ""

    lines = [
        "",
        "━━━ TOOL USE PROTOCOL ━━━",
        "You have access to the following tools. To call a tool, output a",
        "JSON block wrapped in XML tags (one per response, on its own line):",
        "",
        f"  <{TOOL_CALL_TAG}>{{\"name\": \"tool_name\", \"arguments\": {{\"arg\": \"value\"}}}}</{TOOL_CALL_TAG}>",
        "",
        "After each tool call I will return the result wrapped in:",
        f"  <{TOOL_RESULT_TAG}>{{\"tool\": \"...\", \"result\": \"...\"}}</{TOOL_RESULT_TAG}>",
        "",
        "You MUST call tools one at a time and wait for each result.",
        "When you are done with all tool work, give your final text response",
        f"WITHOUT any <{TOOL_CALL_TAG}> tags.",
        "",
        "━━━ AVAILABLE TOOLS ━━━",
    ]

    for schema in tool_schemas:
        fn = schema.get("function", schema)
        name = fn.get("name", "unknown")
        desc = fn.get("description", "")
        params = fn.get("parameters", {}).get("properties", {})
        required = fn.get("parameters", {}).get("required", [])

        lines.append(f"\n  {name}  —  {desc}")
        if params:
            lines.append("  Arguments:")
            for pname, pinfo in params.items():
                req_marker = " [required]" if pname in required else " [optional]"
                pdesc = pinfo.get("description", "")
                ptype = pinfo.get("type", "string")
                lines.append(f"    {pname} ({ptype}){req_marker}: {pdesc}")

    lines.append("\n━━━ END TOOL PROTOCOL ━━━\n")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# ClaudeCliAgent
# ---------------------------------------------------------------------------


class ClaudeCliAgent:
    """
    Lightweight agent backed by the `claude` CLI subprocess.

    Implements the same interface as `core.agent.Agent`:
      result = agent.run(message, context=...)
      result.content, result.tool_calls_made, result.error, ...

    Tool calling uses XML-tagged JSON blocks (see _build_tool_system_section).
    Multi-turn conversation is maintained via --session-id + --resume.
    """

    MAX_TOOL_ROUNDS = 20
    CLI_TIMEOUT_SEC = 120

    def __init__(
        self,
        name: str,
        model: str,
        system_prompt: str,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_handlers: Optional[Dict[str, Callable]] = None,
        temperature: float = 0.3,
        max_tokens: int = 16384,
    ):
        self.name = name
        # strip "claude-cli/" prefix to get bare model name
        self.model = model.split("/", 1)[1] if "/" in model else model
        self.temperature = temperature
        self.max_tokens = max_tokens

        self.tool_handlers = tool_handlers or {}

        # Build augmented system prompt with tool protocol
        tool_section = _build_tool_system_section(tools or [])
        self._system_prompt = system_prompt + tool_section

    def _call_cli(
        self,
        message: str,
        session_id: str,
        first_turn: bool,
    ) -> Tuple[str, float]:
        """
        Call the claude CLI for a single turn.

        Returns (response_text, elapsed_sec).
        Raises RuntimeError if the CLI call fails.
        """
        cmd = [
            CLAUDE_CLI,
            "--print",
            "--output-format", "json",
            "--model", self.model,
        ]

        if first_turn:
            cmd += ["--session-id", session_id, "--system-prompt", self._system_prompt]
        else:
            cmd += ["--resume", session_id]

        cmd.append(message)

        logger.debug(f"[{self.name}] CLI call: model={self.model} first={first_turn}")
        t0 = time.monotonic()

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.CLI_TIMEOUT_SEC,
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"claude CLI timed out after {self.CLI_TIMEOUT_SEC}s")
        except FileNotFoundError:
            raise RuntimeError(
                f"claude CLI not found at {CLAUDE_CLI}. "
                "Install Claude Code: https://claude.ai/code"
            )

        elapsed = time.monotonic() - t0

        if proc.returncode != 0:
            stderr = proc.stderr.strip()
            raise RuntimeError(
                f"claude CLI exited {proc.returncode}: {stderr[:500]}"
            )

        raw = proc.stdout.strip()
        if not raw:
            raise RuntimeError("claude CLI returned empty output")

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"claude CLI output is not JSON: {e}\nRaw: {raw[:500]}")

        if data.get("is_error") or data.get("subtype") == "error":
            raise RuntimeError(f"claude API error: {data.get('result', data)}")

        return data.get("result", ""), elapsed

    def run(
        self,
        message: str,
        context: Optional[str] = None,
    ) -> "AgentResult":
        """
        Execute an agent turn. Loops through tool calls until the model
        produces a final text answer or MAX_TOOL_ROUNDS is hit.
        """
        from core.agent import AgentResult  # local import to avoid circular

        session_id = str(uuid.uuid4())
        total_elapsed = 0.0
        tool_calls_made = 0
        tool_results_log: List[Dict[str, Any]] = []

        # Prepend context as part of the initial message
        if context:
            initial_message = f"{context}\n\n{message}"
        else:
            initial_message = message

        current_message = initial_message
        first_turn = True

        for _round in range(self.MAX_TOOL_ROUNDS):
            try:
                response_text, elapsed = self._call_cli(
                    current_message, session_id, first_turn
                )
            except RuntimeError as e:
                logger.error(f"[{self.name}] CLI error: {e}")
                return AgentResult(
                    error=str(e),
                    tool_calls_made=tool_calls_made,
                    tool_results=tool_results_log,
                    elapsed_sec=total_elapsed,
                )

            total_elapsed += elapsed
            first_turn = False

            # Check for tool calls
            tool_call_matches = _TOOL_CALL_RE.findall(response_text)
            if not tool_call_matches:
                # No tool calls — final response
                return AgentResult(
                    content=response_text,
                    tool_calls_made=tool_calls_made,
                    tool_results=tool_results_log,
                    elapsed_sec=total_elapsed,
                )

            # Execute the first tool call found (one per turn)
            raw_call = tool_call_matches[0]
            try:
                call_data = json.loads(raw_call)
            except json.JSONDecodeError as e:
                # Malformed tool call — feed back error
                current_message = (
                    f"<{TOOL_RESULT_TAG}>{{\"error\": \"Malformed tool_call JSON: {e}\"}}</{TOOL_RESULT_TAG}>"
                )
                continue

            fn_name = call_data.get("name", "")
            fn_args = call_data.get("arguments", {})
            if not isinstance(fn_args, dict):
                fn_args = {}

            tool_calls_made += 1
            handler = self.tool_handlers.get(fn_name)
            if handler is None:
                result_str = json.dumps({"error": f"Unknown tool: {fn_name}"})
            else:
                try:
                    result = handler(**fn_args)
                    if isinstance(result, str):
                        result_str = result
                    else:
                        result_str = json.dumps(result, default=str)
                except Exception as e:
                    logger.warning(f"[{self.name}] Tool {fn_name} failed: {e}")
                    result_str = json.dumps({"error": str(e)})

            tool_results_log.append(
                {
                    "tool": fn_name,
                    "args": fn_args,
                    "result_preview": result_str[:500],
                }
            )

            logger.debug(
                f"[{self.name}] Tool {fn_name}() -> {result_str[:200]}"
            )

            # Feed result back as next user message
            current_message = (
                f"<{TOOL_RESULT_TAG}>"
                + json.dumps({"tool": fn_name, "result": result_str[:4000]})
                + f"</{TOOL_RESULT_TAG}>"
            )

        return AgentResult(
            content="Max tool rounds exceeded.",
            tool_calls_made=tool_calls_made,
            tool_results=tool_results_log,
            elapsed_sec=total_elapsed,
            error="Exceeded MAX_TOOL_ROUNDS",
        )
