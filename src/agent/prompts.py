"""Prompt templates for Control (`raw_diff`) vs. Treatment (`ast_edit`).

Enforces strict experimental control (Proposal Sec. 4.2 & 4.3 + Prof. Shin Yoo Aug 7–8 email):
- Framed accurately as post-fix Bug-Reproducing Test (BRT) generation.
- The system prompt, instructions, tool schemas, and feedback wrapper are
  BYTE-FOR-BYTE IDENTICAL across Control and Treatment.
- ONLY the pre-loaded change representation block switches between:
  - Control (`raw_diff`): unified `git diff`
  - Treatment (`ast_edit`): normalized GumTree AST edit script
"""

from __future__ import annotations

from typing import Literal

from src.ast_diff.hyrep import generate_hyrep_diff

Condition = Literal["raw_diff", "ast_edit", "hyrep"]


SYSTEM_PROMPT = """You are an expert Java software testing engineer.

Your task is to write a single, self-contained JUnit bug-reproducing test (BRT) for a bug-fixing commit.
You are operating against the **pre-fix (buggy)** version of the repository.
A valid bug-reproducing test MUST:
1. Compile cleanly against the pre-fix version.
2. Exercise the buggy behavior introduced/addressed by the commit and **fail on the pre-fix version** with a meaningful JUnit assertion failure (e.g., `assertEquals`, `assertTrue`, `assertFalse`, `assertNotNull`, or an expected-behavior check that fails because the pre-fix code produces the wrong output or throws an unexpected bug exception).
3. Assert the **correct (post-fix) expected behavior**, so that once the fix is applied, the test will **pass on the post-fix version**.

Available tools (all operate strictly on the pre-fix checkout):
- `get_context(rel_path, start_line, end_line)`: Inspect surrounding Java source code in the pre-fix checkout.
- `run_test(package_name, test_class_name, test_method_name, java_source)`: Compile and execute your candidate JUnit test against the pre-fix version and receive compiler output, execution outcome, and diagnostic coverage for the enclosing method.

Rules:
- Put your test in the same Java package as the target class so package-private members are accessible if needed, using the exact test class name provided in the user prompt.
- Use JUnit 4 (`org.junit.Test`, `org.junit.Assert.*`) or JUnit 3 (`junit.framework.TestCase`) matching the project's convention (you can check imports in the target file or existing non-triggering tests).
- Do NOT write trivial tests that fail due to bad setup, missing mocks, or unrelated runtime errors.
- Coverage feedback is purely diagnostic to help you see whether your test reached the enclosing method — do not chase coverage for its own sake.
- Stop as soon as `run_test` reports that your test compiles and fails on the pre-fix version for the right reason.
"""


def build_opening_prompt(
    bug_key: str,
    target_rel_file: str,
    target_class_fqn: str,
    enclosing_method: str,
    enclosing_method_start: int,
    enclosing_method_end: int,
    suggested_test_class: str,
    suggested_test_method: str,
    condition: Condition,
    raw_diff: str,
    ast_edit_script: str,
) -> str:
    """Build the opening user prompt with identical structure across both conditions."""
    package_name = (
        target_class_fqn.rsplit(".", 1)[0] if "." in target_class_fqn else ""
    )

    if condition == "raw_diff":
        rep_label = "Raw Unified Diff (`git diff`)"
        rep_body = f"```diff\n{raw_diff.strip()}\n```"
    elif condition == "ast_edit":
        rep_label = "Normalized AST Edit Script (GumTree)"
        rep_body = f"```text\n{ast_edit_script.strip()}\n```"
    elif condition == "hyrep":
        rep_label = "Hybrid Representation (HyRep-BRT: Unified Diff + AST Semantics)"
        body_text = generate_hyrep_diff(raw_diff, ast_edit_script)
        rep_body = f"```diff\n{body_text.strip()}\n```"
    else:
        raise ValueError(f"Unsupported condition: {condition!r}")

    return f"""## Target Bug Context
- **Bug ID:** `{bug_key}`
- **Modified File (pre-fix):** `{target_rel_file}`
- **Target Class:** `{target_class_fqn}`
- **Java Package:** `{package_name}`
- **Enclosing Method (pre-fix lines {enclosing_method_start}-{enclosing_method_end}):** `{enclosing_method}`
- **Required Test Class Name:** `{suggested_test_class}`
- **Required Test Method Name:** `{suggested_test_method}`

## Bug-Fixing Change Representation ({rep_label})
{rep_body}

## Instructions
1. If you need surrounding pre-fix code, call `get_context` on `{target_rel_file}` (for example lines {max(1, enclosing_method_start - 30)}-{enclosing_method_end + 30}) or related helper classes.
2. Call `run_test` with a complete, compilable Java test class named `{suggested_test_class}` in package `{package_name}` containing `@Test public void {suggested_test_method}()`.
3. Use the compiler output, execution outcome, and diagnostic method coverage from `run_test` to refine your test until it compiles and reproduces the bug on the pre-fix version (up to 5 test-execution attempts).
"""


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_context",
            "description": "Read a bounded line range of a Java source file from the pre-fix checkout.",
            "parameters": {
                "type": "object",
                "properties": {
                    "rel_path": {
                        "type": "string",
                        "description": "Project-relative path inside the pre-fix checkout.",
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "1-indexed inclusive start line.",
                        "default": 1,
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "1-indexed inclusive end line.",
                        "default": 200,
                    },
                },
                "required": ["rel_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_test",
            "description": "Compile and execute a candidate JUnit bug-reproducing test on the pre-fix version.",
            "parameters": {
                "type": "object",
                "properties": {
                    "package_name": {
                        "type": "string",
                        "description": "Java package name (e.g., org.apache.commons.lang3.math).",
                    },
                    "test_class_name": {
                        "type": "string",
                        "description": "Simple Java class name (e.g., AgentBRT_Lang_1_Test).",
                    },
                    "test_method_name": {
                        "type": "string",
                        "description": "JUnit test method name (e.g., testReproduceBug).",
                    },
                    "java_source": {
                        "type": "string",
                        "description": "Full compilable Java source code for the test class.",
                    },
                },
                "required": [
                    "package_name",
                    "test_class_name",
                    "test_method_name",
                    "java_source",
                ],
            },
        },
    },
]
