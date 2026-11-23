"""Security & leakage isolation tests for PrefixSandbox (Proposal Sec. 4.3 + Aug 12 email)."""

from pathlib import Path
import pytest

from src.mcp.sandbox import (
    PrefixSandbox,
    SandboxSecurityError,
    _redact_java_method,
    find_matching_brace,
    strip_triggering_test_methods,
)


def test_sandbox_blocks_traversal_and_triggering_tests(tmp_path: Path):
    raw_pre = tmp_path / "prefix_raw"
    post_dir = tmp_path / "postfix_dir"
    agent_dir = tmp_path / "prefix_agent"

    (raw_pre / "src/main/java/pkg").mkdir(parents=True)
    (raw_pre / "src/test/java/pkg").mkdir(parents=True)
    (post_dir / "src/main/java/pkg").mkdir(parents=True)

    (raw_pre / "src/main/java/pkg/Buggy.java").write_text(
        "package pkg;\npublic class Buggy { int f(int x) { return x - 1; } }\n",
        encoding="utf-8",
    )
    (post_dir / "src/main/java/pkg/Buggy.java").write_text(
        "package pkg;\npublic class Buggy { int f(int x) { return x + 1; } }\n",
        encoding="utf-8",
    )
    (raw_pre / "src/test/java/pkg/BuggyTest.java").write_text(
        "package pkg;\npublic class BuggyTest {\n"
        "  @Test\n"
        "  public void testTrigger() {\n"
        "    assertEquals(2, new Buggy().f(1));\n"
        "  }\n"
        "  @Test\n"
        "  public void testOther() {\n"
        "    assertNotNull(new Buggy());\n"
        "  }\n"
        "}\n",
        encoding="utf-8",
    )

    sandbox = PrefixSandbox.materialize_clean(
        raw_prefix_checkout=raw_pre,
        agent_workspace_dir=agent_dir,
        dir_src_tests="src/test/java",
        target_class_fqn="pkg.Buggy",
        enclosing_method_start=2,
        enclosing_method_end=2,
        triggering_tests=["pkg.BuggyTest::testTrigger"],
        strip_triggers=True,
    )

    # Pre-fix file read works
    ctx = sandbox.get_context("src/main/java/pkg/Buggy.java", 1, 10)
    assert "return x - 1" in ctx

    # Relative path traversal to post-fix fails
    with pytest.raises(SandboxSecurityError):
        sandbox.get_context("../postfix_dir/src/main/java/pkg/Buggy.java", 1, 10)

    # Absolute path to post-fix fails
    with pytest.raises(SandboxSecurityError):
        sandbox.get_context(str(post_dir / "src/main/java/pkg/Buggy.java"), 1, 10)

    # Reading triggering test file fails
    with pytest.raises(SandboxSecurityError):
        sandbox.get_context("src/test/java/pkg/BuggyTest.java", 1, 30)

    # Triggering test method body was physically redacted while other test preserved
    redacted = (agent_dir / "src/test/java/pkg/BuggyTest.java").read_text(encoding="utf-8")
    assert "assertEquals(2" not in redacted
    assert "REDACTED" in redacted
    assert "testOther" in redacted


def test_redact_method_with_braces_in_strings_and_comments():
    source = """package demo;
public class ComplexTest {
    @Test(timeout = 1000)
    public void testTrigger() {
        String s = "nested { brace } and \"quote\" }";
        char c = '}';
        // comment with { and }
        /* block comment
           with { and }
        */
        assertEquals("}", s);
    }
    @Test
    public void testPreserved() {
        assertTrue(true);
    }
}
"""
    redacted = _redact_java_method(source, "testTrigger")
    assert "REDACTED" in redacted
    assert "testPreserved" in redacted
    assert "assertTrue(true);" in redacted
    assert "nested { brace }" not in redacted
    # Verify clean syntax remains
    assert redacted.count("{") == redacted.count("}")
