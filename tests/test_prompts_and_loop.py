"""Tests for prompt symmetry, inline enclosing-method context, and trial loop wiring."""

from pathlib import Path
import os
from typing import Any, Dict, List

from src.agent.llm_client import LLMResponse, ToolCall, _load_dotenv_if_present
from src.agent.loop import run_agentic_trial
from src.agent.prompts import build_opening_prompt
from src.mcp.sandbox import PrefixSandbox


class _DummyLLM:
    model = "dummy-model"

    def __init__(self) -> None:
        self.seen_prompts: List[str] = []

    def chat_with_tools(self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]) -> LLMResponse:
        self.seen_prompts.append(messages[1]["content"])
        return LLMResponse(text="Calling run_test", tool_calls=[], prompt_tokens=10, completion_tokens=5)


def test_opening_prompt_inlines_enclosing_method_symmetrically():
    method_src = "  10 | public int clamp(int x) {\n  11 |   return x;\n  12 | }"
    p_raw = build_opening_prompt(
        bug_key="Lang-99",
        target_rel_file="src/main/java/Demo.java",
        target_class_fqn="org.demo.Demo",
        enclosing_method="clamp",
        enclosing_method_start=10,
        enclosing_method_end=12,
        suggested_test_class="AgentBRT_Lang_99_raw_diff_Test",
        suggested_test_method="testReproduceBug",
        condition="raw_diff",
        raw_diff="- return x;\n+ return Math.max(0, x);",
        ast_edit_script="1. INSERT MethodInvocation",
        enclosing_method_source=method_src,
    )
    p_ast = build_opening_prompt(
        bug_key="Lang-99",
        target_rel_file="src/main/java/Demo.java",
        target_class_fqn="org.demo.Demo",
        enclosing_method="clamp",
        enclosing_method_start=10,
        enclosing_method_end=12,
        suggested_test_class="AgentBRT_Lang_99_ast_edit_Test",
        suggested_test_method="testReproduceBug",
        condition="ast_edit",
        raw_diff="- return x;\n+ return Math.max(0, x);",
        ast_edit_script="1. INSERT MethodInvocation",
        enclosing_method_source=method_src,
    )
    assert "## Enclosing Method Context (Pre-Fix `clamp`, lines 10-12)" in p_raw
    assert "## Enclosing Method Context (Pre-Fix `clamp`, lines 10-12)" in p_ast
    assert "public int clamp(int x)" in p_raw and "public int clamp(int x)" in p_ast
    # Replace only the expected differing sections and verify identical framing
    assert p_raw.split("## Bug-Fixing Change Representation")[0].replace("raw_diff", "COND") == (
        p_ast.split("## Bug-Fixing Change Representation")[0].replace("ast_edit", "COND")
    )


def test_run_agentic_trial_auto_inlines_method_from_sandbox(tmp_path: Path):
    raw_pre = tmp_path / "pre"
    agent_dir = tmp_path / "agent"
    (raw_pre / "src/main/java/org/demo").mkdir(parents=True)
    (raw_pre / "src/test/java/org/demo").mkdir(parents=True)
    (raw_pre / "src/main/java/org/demo/Demo.java").write_text(
        "package org.demo;\npublic class Demo {\n  int f(int x) {\n    return x;\n  }\n}\n",
        encoding="utf-8",
    )
    sandbox = PrefixSandbox.materialize_clean(
        raw_prefix_checkout=raw_pre,
        agent_workspace_dir=agent_dir,
        dir_src_tests="src/test/java",
        target_class_fqn="org.demo.Demo",
        enclosing_method_start=3,
        enclosing_method_end=5,
        triggering_tests=[],
        strip_triggers=False,
    )
    bug = {
        "bug_key": "Demo-1",
        "modified_classes": ["org.demo.Demo"],
        "target_rel_file": "src/main/java/org/demo/Demo.java",
        "enclosing_method": "f",
        "enclosing_method_start": 3,
        "enclosing_method_end": 5,
        "raw_diff": "- return x;\n+ return x + 1;",
        "ast_edit_script": "1. INSERT InfixExpression :: `x + 1`",
    }
    llm = _DummyLLM()
    run_agentic_trial(
        bug=bug,
        condition="ast_edit",
        sandbox=sandbox,
        llm=llm,
        max_retries=1,
        max_tool_calls_per_retry=1,
        inline_enclosing_method=True,
    )
    assert llm.seen_prompts
    assert "int f(int x)" in llm.seen_prompts[0]
    assert "return x;" in llm.seen_prompts[0]


def test_load_dotenv_if_present(tmp_path: Path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("# comment\nTEST_DUMMY_KEY_XYZ=abc123\n", encoding="utf-8")
    monkeypatch.delenv("TEST_DUMMY_KEY_XYZ", raising=False)
    _load_dotenv_if_present(env_file)
    assert os.environ.get("TEST_DUMMY_KEY_XYZ") == "abc123"
    # Existing env var is preserved
    env_file.write_text("TEST_DUMMY_KEY_XYZ=other\n", encoding="utf-8")
    _load_dotenv_if_present(env_file)
    assert os.environ.get("TEST_DUMMY_KEY_XYZ") == "abc123"
