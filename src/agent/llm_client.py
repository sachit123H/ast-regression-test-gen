"""Provider-agnostic LLM client with native function/tool calling and token tracking.

Supports:
- OpenAI (`provider: "openai"`)
- Any OpenAI-compatible endpoint (`provider: "openai_compatible"`, e.g. Gemini OpenAI-compat, vLLM, OpenRouter)
- Anthropic Claude (`provider: "anthropic"`)
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


def _load_dotenv_if_present(dotenv_path: Path = Path(".env")) -> None:
    """Populate missing env vars from a local gitignored `.env` file if present."""
    if not dotenv_path.is_file():
        return
    for raw_line in dotenv_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


def _retry_with_backoff(
    fn: Callable[[], Any],
    max_retries: int = 6,
    initial_delay: float = 2.0,
    backoff_factor: float = 2.0,
    max_delay: float = 60.0,
) -> Any:
    """Execute `fn` with exponential backoff + jitter on transient rate-limit/network errors."""
    import random

    delay = initial_delay
    last_exc: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            exc_str = str(exc).lower()
            # Never retry permanent billing, quota-exhaustion, or authentication failures
            is_permanent_billing_or_auth = any(
                tok in exc_str
                for tok in (
                    "your account is not active",
                    "insufficient_quota",
                    "billing",
                    "invalid_api_key",
                    "authentication",
                    "unauthorized",
                )
            )
            is_transient = (not is_permanent_billing_or_auth) and any(
                tok in exc_str
                for tok in (
                    "rate_limit",
                    "ratelimit",
                    "429",
                    "500",
                    "502",
                    "503",
                    "504",
                    "timeout",
                    "connection",
                    "overloaded",
                )
            )
            if attempt == max_retries or not is_transient:
                raise
            sleep_sec = min(max_delay, delay) + random.uniform(0.0, 1.0)
            time.sleep(sleep_sec)
            delay = min(max_delay, delay * backoff_factor)
    raise last_exc  # type: ignore


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass
class LLMResponse:
    text: str
    tool_calls: List[ToolCall] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0


class LLMClient:
    """Thin wrapper normalizing OpenAI and Anthropic tool-calling APIs."""

    def __init__(
        self,
        provider: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        api_base_url: Optional[str] = None,
    ) -> None:
        self.provider = provider.lower()
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_base_url = api_base_url

        if self.provider in {"openai", "openai_compatible"}:
            from openai import OpenAI  # type: ignore

            _load_dotenv_if_present()
            kwargs: Dict[str, Any] = {}
            if api_base_url:
                kwargs["base_url"] = api_base_url
            api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("LLM_API_KEY")
            if api_key:
                kwargs["api_key"] = api_key
            self._client = OpenAI(**kwargs)
        elif self.provider == "anthropic":
            from anthropic import Anthropic  # type: ignore

            self._client = Anthropic()
        else:
            raise ValueError(f"Unsupported LLM provider: {provider!r}")

    def chat_with_tools(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
    ) -> LLMResponse:
        """Execute a single chat completion step with tool schemas."""
        if self.provider in {"openai", "openai_compatible"}:
            return self._call_openai(messages, tools)
        return self._call_anthropic(messages, tools)

    def _call_openai(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
    ) -> LLMResponse:
        def _invoke():
            return self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

        resp = _retry_with_backoff(_invoke)
        msg = resp.choices[0].message
        tool_calls: List[ToolCall] = []
        for tc in msg.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args))

        usage = getattr(resp, "usage", None)
        return LLMResponse(
            text=msg.content or "",
            tool_calls=tool_calls,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) if usage else 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) if usage else 0,
        )

    def _call_anthropic(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
    ) -> LLMResponse:
        system_prompt = ""
        anth_messages: List[Dict[str, Any]] = []
        for m in messages:
            role = m["role"]
            if role == "system":
                system_prompt = m["content"]
            elif role == "tool":
                tool_result_block = {
                    "type": "tool_result",
                    "tool_use_id": m["tool_call_id"],
                    "content": str(m["content"]),
                }
                # Merge consecutive tool results into a single user message (Anthropic requirement)
                if (
                    anth_messages
                    and anth_messages[-1]["role"] == "user"
                    and isinstance(anth_messages[-1]["content"], list)
                    and all(isinstance(b, dict) and b.get("type") == "tool_result" for b in anth_messages[-1]["content"])
                ):
                    anth_messages[-1]["content"].append(tool_result_block)
                else:
                    anth_messages.append({"role": "user", "content": [tool_result_block]})
            elif role == "assistant" and m.get("tool_calls"):
                blocks: List[Dict[str, Any]] = []
                if m.get("content"):
                    blocks.append({"type": "text", "text": m["content"]})
                for tc in m["tool_calls"]:
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": tc["id"],
                            "name": tc["function"]["name"],
                            "input": json.loads(tc["function"]["arguments"]),
                        }
                    )
                anth_messages.append({"role": "assistant", "content": blocks})
            else:
                anth_messages.append({"role": role, "content": m["content"]})

        anth_tools = [
            {
                "name": t["function"]["name"],
                "description": t["function"]["description"],
                "input_schema": t["function"]["parameters"],
            }
            for t in tools
        ]

        def _invoke():
            return self._client.messages.create(
                model=self.model,
                system=system_prompt,
                messages=anth_messages,
                tools=anth_tools,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

        resp = _retry_with_backoff(_invoke)
        text_parts: List[str] = []
        tool_calls: List[ToolCall] = []
        for block in resp.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(id=block.id, name=block.name, arguments=dict(block.input))
                )
        return LLMResponse(
            text="\n".join(text_parts),
            tool_calls=tool_calls,
            prompt_tokens=getattr(resp.usage, "input_tokens", 0),
            completion_tokens=getattr(resp.usage, "output_tokens", 0),
        )
