"""Closed 5-retry agentic loop for bug-reproducing test generation.

Implements Proposal Section 4.2 & 4.3:
- Pre-loads either raw diff (Control) or normalized AST edit script (Treatment)
  into the opening prompt before the loop starts.
- Routes tool calls (`get_context`, `run_test`) strictly through `PrefixSandbox`
  (or via MCP stdio transport) so post-fix code and triggering tests are unreachable.
- Stops immediately when a candidate test compiles AND fails on the pre-fix version,
  or when `max_retries` (default 5) `run_test` attempts are exhausted.
- Logs full attempt trace, token usage, latency, and final candidate test to JSON.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from src.agent.llm_client import LLMClient
from src.agent.prompts import (
    Condition,
    SYSTEM_PROMPT,
    TOOL_SCHEMAS,
    build_opening_prompt,
)
from src.mcp.sandbox import PrefixSandbox, SandboxSecurityError, TestRunFeedback


@dataclass
class AttemptRecord:
    attempt_index: int
    package_name: str
    test_class_name: str
    test_method_name: str
    java_source: str
    compiled: bool
    passed_on_prefix: Optional[bool]
    failure_type: str
    exception_type: str
    exception_message: str
    enclosing_method_hit: bool
    enclosing_method_covered_lines: List[int]
    enclosing_method_missed_lines: List[int]


@dataclass
class AgentTrialResult:
    bug_key: str
    condition: Condition
    model: str
    terminated_reason: str            # PREFIX_FAIL_REACHED | MAX_RETRIES_EXHAUSTED | NO_TOOL_CALL
    num_run_test_attempts: int
    num_get_context_calls: int
    prompt_tokens: int
    completion_tokens: int
    wall_time_sec: float
    final_package_name: str = ""
    final_test_class_name: str = ""
    final_test_method_name: str = ""
    final_java_source: str = ""
    final_prefix_failure_type: str = ""
    attempts: List[AttemptRecord] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def run_agentic_trial(
    bug: Dict[str, Any],
    condition: Condition,
    sandbox: PrefixSandbox,
    llm: LLMClient,
    max_retries: int = 5,
    max_tool_calls_per_retry: int = 6,
) -> AgentTrialResult:
    """Execute one closed agentic BRT generation trial for `(bug, condition)`."""
    t0 = time.monotonic()
    bug_key = bug["bug_key"]
    safe_key = bug_key.replace("-", "_")
    target_class_fqn = bug["modified_classes"][0]
    suggested_class = f"AgentBRT_{safe_key}_{condition}_Test"
    suggested_method = "testReproduceBug"

    opening_prompt = build_opening_prompt(
        bug_key=bug_key,
        target_rel_file=bug["target_rel_file"],
        target_class_fqn=target_class_fqn,
        enclosing_method=bug["enclosing_method"],
        enclosing_method_start=int(bug["enclosing_method_start"]),
        enclosing_method_end=int(bug["enclosing_method_end"]),
        suggested_test_class=suggested_class,
        suggested_test_method=suggested_method,
        condition=condition,
        raw_diff=bug["raw_diff"],
        ast_edit_script=bug["ast_edit_script"],
    )

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": opening_prompt},
    ]

    attempts: List[AttemptRecord] = []
    num_context_calls = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    terminated_reason = "MAX_RETRIES_EXHAUSTED"
    max_steps = max_retries * max_tool_calls_per_retry

    for _step in range(max_steps):
        if len(attempts) >= max_retries:
            terminated_reason = "MAX_RETRIES_EXHAUSTED"
            break

        resp = llm.chat_with_tools(messages=messages, tools=TOOL_SCHEMAS)
        total_prompt_tokens += resp.prompt_tokens
        total_completion_tokens += resp.completion_tokens

        if not resp.tool_calls:
            # Prompt agent once if it returned code in prose instead of calling run_test
            messages.append({"role": "assistant", "content": resp.text})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Please invoke the `run_test` tool with your complete Java test class "
                        "so it can be compiled and executed on the pre-fix version."
                    ),
                }
            )
            continue

        messages.append(
            {
                "role": "assistant",
                "content": resp.text,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                    }
                    for tc in resp.tool_calls
                ],
            }
        )

        stop_loop = False
        for tc in resp.tool_calls:
            if tc.name == "get_context":
                num_context_calls += 1
                try:
                    out = sandbox.get_context(
                        rel_path=str(tc.arguments.get("rel_path", "")),
                        start_line=int(tc.arguments.get("start_line", 1)),
                        end_line=int(tc.arguments.get("end_line", 200)),
                    )
                except SandboxSecurityError as exc:
                    out = f"SECURITY_ERROR: {exc}"
                messages.append(
                    {"role": "tool", "tool_call_id": tc.id, "content": out}
                )

            elif tc.name == "run_test":
                pkg = str(tc.arguments.get("package_name") or "")
                cls_name = str(tc.arguments.get("test_class_name") or suggested_class)
                m_name = str(tc.arguments.get("test_method_name") or suggested_method)
                java_src = str(tc.arguments.get("java_source") or "")

                try:
                    fb: TestRunFeedback = sandbox.run_test(
                        package_name=pkg,
                        test_class_name=cls_name,
                        test_method_name=m_name,
                        java_source=java_src,
                    )
                    fb_text = fb.to_agent_text()
                except SandboxSecurityError as exc:
                    fb = TestRunFeedback(
                        compiled=False,
                        compiler_output=f"SECURITY_ERROR: {exc}",
                        passed=None,
                        failure_type="COMPILE_ERROR",
                        exception_type="SandboxSecurityError",
                        exception_message=str(exc),
                        stack_trace="",
                        enclosing_method_covered_lines=[],
                        enclosing_method_missed_lines=[],
                        enclosing_method_hit=False,
                    )
                    fb_text = fb.to_agent_text()

                attempt_rec = AttemptRecord(
                    attempt_index=len(attempts) + 1,
                    package_name=pkg,
                    test_class_name=cls_name,
                    test_method_name=m_name,
                    java_source=java_src,
                    compiled=fb.compiled,
                    passed_on_prefix=fb.passed,
                    failure_type=fb.failure_type,
                    exception_type=fb.exception_type,
                    exception_message=fb.exception_message,
                    enclosing_method_hit=fb.enclosing_method_hit,
                    enclosing_method_covered_lines=fb.enclosing_method_covered_lines,
                    enclosing_method_missed_lines=fb.enclosing_method_missed_lines,
                )
                attempts.append(attempt_rec)
                messages.append(
                    {"role": "tool", "tool_call_id": tc.id, "content": fb_text}
                )

                # Proposal Sec. 4.2 stopping criterion: compiles AND fails on pre-fix
                if fb.compiled and fb.passed is False and fb.failure_type != "TIMEOUT":
                    terminated_reason = "PREFIX_FAIL_REACHED"
                    stop_loop = True
                    break

                if len(attempts) >= max_retries:
                    terminated_reason = "MAX_RETRIES_EXHAUSTED"
                    stop_loop = True
                    break
            else:
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": f"Unknown tool: {tc.name}",
                    }
                )

        if stop_loop:
            break

    if not attempts:
        terminated_reason = "NO_TOOL_CALL"
    last = attempts[-1] if attempts else None
    return AgentTrialResult(
        bug_key=bug_key,
        condition=condition,
        model=llm.model,
        terminated_reason=terminated_reason,
        num_run_test_attempts=len(attempts),
        num_get_context_calls=num_context_calls,
        prompt_tokens=total_prompt_tokens,
        completion_tokens=total_completion_tokens,
        wall_time_sec=round(time.monotonic() - t0, 3),
        final_package_name=last.package_name if last else "",
        final_test_class_name=last.test_class_name if last else "",
        final_test_method_name=last.test_method_name if last else "",
        final_java_source=last.java_source if last else "",
        final_prefix_failure_type=last.failure_type if last else "NO_ATTEMPT",
        attempts=attempts,
    )
