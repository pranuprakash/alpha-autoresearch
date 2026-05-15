"""
LLM Agent — multi-provider routing. No litellm, no framework.

Model prefix routing:
  claude-cli/* → local `claude` CLI subprocess (no API key, uses Max subscription)
  openai/*     → api.openai.com          (OPENAI_API_KEY)
  anthropic/*  → api.anthropic.com/v1   (ANTHROPIC_API_KEY)
  google/*     → generativelanguage.googleapis.com/v1beta/openai/ (GOOGLE_API_KEY)
  fireworks/*  → api.fireworks.ai/inference/v1 (FIREWORKS_API_KEY)

DO NOT add litellm as a dependency. It had a supply-chain compromise.
Default model (config.yaml) is claude-cli/claude-sonnet-4-6 — no API key needed.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from openai import OpenAI

logger = logging.getLogger("Agent")


def _make_client(model: str) -> Tuple[OpenAI, str]:
    """Return (OpenAI client, bare model name) for a provider-prefixed model string."""
    if "/" not in model:
        return OpenAI(api_key=os.environ["OPENAI_API_KEY"]), model

    provider, bare_model = model.split("/", 1)

    if provider == "openai":
        return OpenAI(api_key=os.environ["OPENAI_API_KEY"]), bare_model

    elif provider == "anthropic":
        return OpenAI(
            api_key=os.environ["ANTHROPIC_API_KEY"],
            base_url="https://api.anthropic.com/v1",
            default_headers={"anthropic-version": "2023-06-01"},
        ), bare_model

    elif provider == "google":
        return OpenAI(
            api_key=os.environ["GOOGLE_API_KEY"],
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        ), bare_model

    elif provider == "fireworks":
        return OpenAI(
            api_key=os.environ.get("FIREWORKS_API_KEY", ""),
            base_url="https://api.fireworks.ai/inference/v1",
        ), bare_model

    else:
        raise ValueError(f"Unsupported model provider: {provider!r} (model={model!r})")


@dataclass
class AgentResult:
    content: str = ""
    tool_calls_made: int = 0
    tool_results: List[Dict[str, Any]] = field(default_factory=list)
    usage: Dict[str, int] = field(default_factory=dict)
    cost_usd: float = 0.0
    elapsed_sec: float = 0.0
    error: Optional[str] = None


def _is_claude_cli_model(model: str) -> bool:
    """Return True if the model string requests the local CLI backend."""
    return model.startswith("claude-cli/") or model.startswith("claude-cli:")


class Agent:
    """
    Lightweight LLM agent with tool-use loop.

    Routes automatically:
    - claude-cli/* → ClaudeCliAgent (local subprocess, no API key)
    - any other prefix → OpenAI-compatible API (requires API key in .env)

    Callers use the same interface regardless of backend.
    """

    MAX_TOOL_ROUNDS = 20

    def __new__(
        cls,
        name: str,
        model: str,
        system_prompt: str,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_handlers: Optional[Dict[str, Callable]] = None,
        temperature: float = 0.3,
        max_tokens: int = 16384,
    ):
        if _is_claude_cli_model(model):
            from core.claude_cli import ClaudeCliAgent
            return ClaudeCliAgent(
                name=name,
                model=model,
                system_prompt=system_prompt,
                tools=tools,
                tool_handlers=tool_handlers,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        obj = super().__new__(cls)
        return obj

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
        if _is_claude_cli_model(model):
            return  # already initialized by __new__ → ClaudeCliAgent
        self.name = name
        self.model = model
        self.system_prompt = system_prompt
        self.tools = tools or []
        self.tool_handlers = tool_handlers or {}
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client, self._bare_model = _make_client(model)

    def run(self, message: str, context: Optional[str] = None) -> AgentResult:
        """
        Execute an agent turn. Loops through tool calls until the LLM
        produces a final text answer or MAX_TOOL_ROUNDS is hit.
        """
        start = time.monotonic()

        messages = [{"role": "system", "content": self.system_prompt}]
        if context:
            messages.append({"role": "user", "content": context})
        messages.append({"role": "user", "content": message})

        total_usage = {"prompt_tokens": 0, "completion_tokens": 0}
        total_cost = 0.0
        tool_calls_made = 0
        tool_results = []

        for _round in range(self.MAX_TOOL_ROUNDS):
            try:
                kwargs: Dict[str, Any] = {
                    "model": self._bare_model,
                    "messages": messages,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                }
                if self.tools:
                    kwargs["tools"] = self.tools
                    kwargs["tool_choice"] = "auto"

                response = self._client.chat.completions.create(**kwargs)
            except Exception as e:
                logger.error(f"[{self.name}] LLM call failed: {e}")
                return AgentResult(
                    error=str(e),
                    elapsed_sec=time.monotonic() - start,
                )

            choice = response.choices[0]
            message_obj = choice.message

            if response.usage:
                total_usage["prompt_tokens"] += response.usage.prompt_tokens or 0
                total_usage["completion_tokens"] += response.usage.completion_tokens or 0

            messages.append(message_obj.model_dump())

            if not message_obj.tool_calls:
                return AgentResult(
                    content=message_obj.content or "",
                    tool_calls_made=tool_calls_made,
                    tool_results=tool_results,
                    usage=total_usage,
                    cost_usd=total_cost,
                    elapsed_sec=time.monotonic() - start,
                )

            for tc in message_obj.tool_calls:
                tool_calls_made += 1
                fn_name = tc.function.name
                try:
                    fn_args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    fn_args = {}

                handler = self.tool_handlers.get(fn_name)
                if handler is None:
                    result_str = json.dumps({"error": f"Unknown tool: {fn_name}"})
                else:
                    try:
                        result = handler(**fn_args)
                        result_str = json.dumps(result, default=str) if not isinstance(result, str) else result
                    except Exception as e:
                        logger.warning(f"[{self.name}] Tool {fn_name} failed: {e}")
                        result_str = json.dumps({"error": str(e)})

                tool_results.append({
                    "tool": fn_name,
                    "args": fn_args,
                    "result_preview": result_str[:500],
                })

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result_str,
                })

                logger.debug(f"[{self.name}] Tool {fn_name}() -> {result_str[:200]}")

        return AgentResult(
            content="Max tool rounds exceeded.",
            tool_calls_made=tool_calls_made,
            tool_results=tool_results,
            usage=total_usage,
            cost_usd=total_cost,
            elapsed_sec=time.monotonic() - start,
            error="Exceeded MAX_TOOL_ROUNDS",
        )


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load config.yaml and resolve model names."""
    import yaml
    with open(config_path) as f:
        return yaml.safe_load(f)
