"""Model Context Protocol (MCP) server exposing ONLY `get_context` and `run_test`.

Implements Proposal Section 4.3:
- Operates strictly over a pre-fix Defects4J checkout root passed via environment
  variables (no version parameter, no path argument escaping the pre-fix root).
- Enforces triggering-test isolation via `PrefixSandbox`.
- Communicates over stdio using the official Python `mcp` SDK (`FastMCP`).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Set

from src.mcp.sandbox import PrefixSandbox, SandboxSecurityError


def build_mcp_server(sandbox: PrefixSandbox):
    """Create a FastMCP server bound to a single isolated `PrefixSandbox`."""
    try:
        from mcp.server.fastmcp import FastMCP  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "The `mcp` Python SDK is required to run the MCP server (`pip install mcp`)."
        ) from exc

    mcp = FastMCP("defects4j-prefix-brt-server")

    @mcp.tool()
    def get_context(rel_path: str, start_line: int = 1, end_line: int = 200) -> str:
        """Read surrounding source code from the pre-fix checkout only.

        Args:
            rel_path: Project-relative file path inside the pre-fix checkout.
            start_line: 1-indexed start line (inclusive).
            end_line: 1-indexed end line (inclusive).
        """
        try:
            return sandbox.get_context(rel_path=rel_path, start_line=start_line, end_line=end_line)
        except SandboxSecurityError as exc:
            return f"SECURITY_ERROR: {exc}"

    @mcp.tool()
    def run_test(
        package_name: str,
        test_class_name: str,
        test_method_name: str,
        java_source: str,
    ) -> str:
        """Compile and execute a candidate JUnit bug-reproducing test on the pre-fix version.

        Returns verbatim compiler output, pre-fix execution outcome, and diagnostic
        enclosing-method coverage.
        """
        try:
            fb = sandbox.run_test(
                package_name=package_name,
                test_class_name=test_class_name,
                test_method_name=test_method_name,
                java_source=java_source,
            )
            return json.dumps(
                {
                    "compiled": fb.compiled,
                    "passed": fb.passed,
                    "failure_type": fb.failure_type,
                    "exception_type": fb.exception_type,
                    "exception_message": fb.exception_message,
                    "enclosing_method_hit": fb.enclosing_method_hit,
                    "enclosing_method_covered_lines": fb.enclosing_method_covered_lines,
                    "enclosing_method_missed_lines": fb.enclosing_method_missed_lines,
                    "feedback_text": fb.to_agent_text(),
                },
                indent=2,
            )
        except SandboxSecurityError as exc:
            return json.dumps({"compiled": False, "error": f"SECURITY_ERROR: {exc}"})

    return mcp


def main() -> None:
    """Launch stdio MCP server configured from environment variables."""
    root = Path(os.environ["BRT_PREFIX_ROOT"])
    dir_src_tests = os.environ.get("BRT_DIR_SRC_TESTS", "src/test/java")
    target_class = os.environ["BRT_TARGET_CLASS"]
    start_line = int(os.environ.get("BRT_METHOD_START", "1"))
    end_line = int(os.environ.get("BRT_METHOD_END", "1"))
    blocked_raw = os.environ.get("BRT_BLOCKED_PATHS", "[]")
    blocked: Set[str] = set(json.loads(blocked_raw))

    sandbox = PrefixSandbox(
        workspace_root=root,
        dir_src_tests=dir_src_tests,
        target_class_fqn=target_class,
        enclosing_method_start=start_line,
        enclosing_method_end=end_line,
        blocked_rel_paths=blocked,
        d4j_bin=os.environ.get("BRT_D4J_BIN", "defects4j"),
    )
    server = build_mcp_server(sandbox)
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
