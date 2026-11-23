"""Isolated Differential Execution Oracle (Proposal Sec. 4.5).

Applied ONCE per bug AFTER the agentic loop terminates; its result is NEVER
returned to the agent. Operates on a separate post-fix directory owned strictly
by the evaluation harness.

Outcome classification:
- `VALID_ASSERTION`  : Fails on pre-fix due to JUnit assertion failure AND passes on post-fix.
- `VALID_EXCEPTION`  : Fails on pre-fix due to runtime exception AND passes on post-fix (LIBRO criterion).
- `FAIL_FAIL`        : Fails on pre-fix AND fails on post-fix (assertion not satisfied by fix).
- `PREFIX_PASS`      : Passed on pre-fix (did not meet loop stopping criterion).
- `COMPILE_ERROR`    : Failed to compile on pre-fix or post-fix.
- `NO_CANDIDATE`     : Agent produced no compilable test.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict

from src.mcp.sandbox import classify_junit_failure, get_d4j_env, wrap_d4j_cmd


@dataclass
class OracleVerdict:
    bug_key: str
    condition: str
    outcome_pair: str               # e.g., "fail/pass", "fail/fail", "pass/n_a", "compile_error"
    classification: str             # VALID_ASSERTION | VALID_EXCEPTION | FAIL_FAIL | PREFIX_PASS | COMPILE_ERROR | NO_CANDIDATE
    is_valid_strict_assertion: bool # Proposal Sec. 4.5 strict assertion-only validity
    is_valid_libro: bool            # LIBRO fail-on-buggy / pass-on-fixed validity
    prefix_failure_type: str
    postfix_compiled: bool
    postfix_passed: bool
    postfix_failure_details: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def evaluate_differential_oracle(
    bug: Dict[str, Any],
    trial: Dict[str, Any],
    postfix_checkout_dir: Path,
    d4j_bin: str = "defects4j",
    compile_timeout: int = 120,
    test_timeout: int = 120,
) -> OracleVerdict:
    """Validate a terminated agent trial against the isolated post-fix checkout."""
    bug_key = bug["bug_key"]
    condition = trial["condition"]
    java_src = trial.get("final_java_source", "")
    prefix_fail_type = trial.get("final_prefix_failure_type", "NO_ATTEMPT")
    terminated_reason = trial.get("terminated_reason", "")

    if not java_src or terminated_reason != "PREFIX_FAIL_REACHED":
        cls_name = "PREFIX_PASS" if prefix_fail_type == "PASS" else (
            "COMPILE_ERROR" if prefix_fail_type == "COMPILE_ERROR" else "NO_CANDIDATE"
        )
        return OracleVerdict(
            bug_key=bug_key,
            condition=condition,
            outcome_pair="pass/n_a" if cls_name == "PREFIX_PASS" else "invalid/n_a",
            classification=cls_name,
            is_valid_strict_assertion=False,
            is_valid_libro=False,
            prefix_failure_type=prefix_fail_type,
            postfix_compiled=False,
            postfix_passed=False,
        )

    pkg = trial["final_package_name"]
    test_cls = trial["final_test_class_name"]
    test_method = trial["final_test_method_name"]
    dir_src_tests = bug["dir_src_tests"]

    # Materialize clean temporary copy of post-fix checkout to keep base checkout pristine
    eval_work_dir = postfix_checkout_dir.parent / f"{bug_key}_eval_{condition}"
    if eval_work_dir.exists():
        shutil.rmtree(eval_work_dir)
    shutil.copytree(postfix_checkout_dir, eval_work_dir, symlinks=False)

    try:
        pkg_path = pkg.replace(".", "/") if pkg else ""
        rel_test = (
            f"{dir_src_tests}/{pkg_path}/{test_cls}.java"
            if pkg_path
            else f"{dir_src_tests}/{test_cls}.java"
        )
        test_file = eval_work_dir / rel_test
        test_file.parent.mkdir(parents=True, exist_ok=True)
        test_file.write_text(java_src, encoding="utf-8")

        is_chart = bug.get("project") == "Chart" or "org.jfree" in bug.get("target_rel_file", "")
        d4j_env = get_d4j_env()

        comp = subprocess.run(
            wrap_d4j_cmd([d4j_bin, "compile"], is_chart=is_chart),
            cwd=str(eval_work_dir),
            capture_output=True,
            text=True,
            errors="replace",
            timeout=compile_timeout,
            env=d4j_env,
        )
        if comp.returncode != 0:
            return OracleVerdict(
                bug_key=bug_key,
                condition=condition,
                outcome_pair="fail/compile_error",
                classification="COMPILE_ERROR",
                is_valid_strict_assertion=False,
                is_valid_libro=False,
                prefix_failure_type=prefix_fail_type,
                postfix_compiled=False,
                postfix_passed=False,
                postfix_failure_details=(comp.stdout + "\n" + comp.stderr)[:1000],
            )

        fqn_test = f"{pkg}.{test_cls}::{test_method}" if pkg else f"{test_cls}::{test_method}"
        run_proc = subprocess.run(
            wrap_d4j_cmd([d4j_bin, "test", "-t", fqn_test], is_chart=is_chart),
            cwd=str(eval_work_dir),
            capture_output=True,
            text=True,
            errors="replace",
            timeout=test_timeout,
            env=d4j_env,
        )
        postfix_passed = run_proc.returncode == 0 and "Failing tests: 0" in run_proc.stdout
        if postfix_passed:
            is_strict = prefix_fail_type == "ASSERTION_FAILURE"
            return OracleVerdict(
                bug_key=bug_key,
                condition=condition,
                outcome_pair="fail/pass",
                classification="VALID_ASSERTION" if is_strict else "VALID_EXCEPTION",
                is_valid_strict_assertion=is_strict,
                is_valid_libro=True,
                prefix_failure_type=prefix_fail_type,
                postfix_compiled=True,
                postfix_passed=True,
            )

        failing_log_path = eval_work_dir / "failing_tests"
        failing_log = (
            failing_log_path.read_text(encoding="utf-8", errors="replace")
            if failing_log_path.exists()
            else (run_proc.stdout + "\n" + run_proc.stderr)
        )
        _, exc_type, exc_msg, _ = classify_junit_failure(failing_log, test_cls)
        return OracleVerdict(
            bug_key=bug_key,
            condition=condition,
            outcome_pair="fail/fail",
            classification="FAIL_FAIL",
            is_valid_strict_assertion=False,
            is_valid_libro=False,
            prefix_failure_type=prefix_fail_type,
            postfix_compiled=True,
            postfix_passed=False,
            postfix_failure_details=f"{exc_type}: {exc_msg}",
        )
    finally:
        if eval_work_dir.exists():
            shutil.rmtree(eval_work_dir, ignore_errors=True)
