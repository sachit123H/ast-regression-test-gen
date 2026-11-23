"""Automated isolation & leakage verification audit (Proposal Sec. 4.3 + Aug 12 email).

Proves programmatically before running experiments that:
1. Post-fix directory is unreachable via relative traversal (`../`) or absolute paths.
2. Pre-existing Defects4J triggering tests are both physically redacted AND blocked
   by `PrefixSandbox.get_context()`.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.mcp.sandbox import PrefixSandbox, SandboxSecurityError


def run_audit() -> bool:
    with tempfile.TemporaryDirectory(prefix="isolation_audit_") as tmp:
        base = Path(tmp)
        raw_pre = base / "prefix_raw"
        post_dir = base / "postfix_secret"
        agent_dir = base / "prefix_agent"

        (raw_pre / "src/main/java/demo").mkdir(parents=True)
        (raw_pre / "src/test/java/demo").mkdir(parents=True)
        (post_dir / "src/main/java/demo").mkdir(parents=True)

        (raw_pre / "src/main/java/demo/Calc.java").write_text(
            "package demo;\npublic class Calc { int add(int a, int b) { return a - b; } }\n",
            encoding="utf-8",
        )
        (post_dir / "src/main/java/demo/Calc.java").write_text(
            "package demo;\npublic class Calc { int add(int a, int b) { return a + b; } }\n",
            encoding="utf-8",
        )
        (raw_pre / "src/test/java/demo/CalcTest.java").write_text(
            "package demo;\npublic class CalcTest {\n"
            "  @Test public void testBug() { assertEquals(4, new Calc().add(2, 2)); }\n"
            "}\n",
            encoding="utf-8",
        )

        sandbox = PrefixSandbox.materialize_clean(
            raw_prefix_checkout=raw_pre,
            agent_workspace_dir=agent_dir,
            dir_src_tests="src/test/java",
            target_class_fqn="demo.Calc",
            enclosing_method_start=2,
            enclosing_method_end=2,
            triggering_tests=["demo.CalcTest::testBug"],
            strip_triggers=True,
        )

        # Check 1: Valid pre-fix read succeeds
        ok_ctx = sandbox.get_context("src/main/java/demo/Calc.java", 1, 10)
        assert "return a - b" in ok_ctx

        # Check 2: Path traversal to post-fix blocked
        try:
            sandbox.get_context("../postfix_secret/src/main/java/demo/Calc.java", 1, 10)
            raise AssertionError("Path traversal was NOT blocked!")
        except SandboxSecurityError:
            pass

        # Check 3: Absolute path to post-fix blocked
        try:
            sandbox.get_context(str(post_dir / "src/main/java/demo/Calc.java"), 1, 10)
            raise AssertionError("Absolute path was NOT blocked!")
        except SandboxSecurityError:
            pass

        # Check 4: Triggering test blocked by denylist AND physically redacted
        try:
            sandbox.get_context("src/test/java/demo/CalcTest.java", 1, 20)
            raise AssertionError("Triggering test file read was NOT blocked!")
        except SandboxSecurityError:
            pass

        redacted_text = (agent_dir / "src/test/java/demo/CalcTest.java").read_text(encoding="utf-8")
        assert "assertEquals(4" not in redacted_text
        assert "REDACTED" in redacted_text

    print("ALL ISOLATION & LEAKAGE CHECKS PASSED.")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify MCP sandbox isolation guarantees.")
    parser.add_argument("--config", default="configs/experiment.yaml")
    _ = parser.parse_args()
    run_audit()


if __name__ == "__main__":
    main()
