"""Hardened pre-fix sandbox for MCP `get_context` and `run_test` tools.

Enforces TWO critical isolation guarantees programmatically (never via prompt instructions):
1. Post-fix isolation (Proposal Sec. 4.3):
   - Operates strictly inside a dedicated pre-fix workspace root.
   - Rejects path traversal (`..`), absolute paths, and symlinks resolving outside root.
   - Exposes NO version parameter.
2. Triggering-test leakage isolation (Sachit's Aug 12, 2026 email to Prof. Shin Yoo):
   - Physically strips/neutralizes Defects4J's pre-existing `tests.trigger` methods
     when materializing the agent's pre-fix workspace.
   - Enforces a strict path denylist in `get_context` blocking any read of triggering
     test classes so the agent cannot peek at ground-truth BRTs.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


def _resolve_utf8_locale() -> str:
    """Return `en_US.UTF-8` if generated in the OS, otherwise fall back to `C.UTF-8`."""
    try:
        proc = subprocess.run(["locale", "-a"], capture_output=True, text=True, timeout=5)
        if proc.returncode == 0:
            avail = {ln.strip().lower() for ln in proc.stdout.splitlines() if ln.strip()}
            if "en_us.utf8" in avail or "en_us.utf-8" in avail:
                return "en_US.UTF-8"
            if "c.utf8" in avail or "c.utf-8" in avail:
                return "C.UTF-8"
    except Exception:
        pass
    return "C.UTF-8"


_CACHED_UTF8_LOCALE: Optional[str] = None


def get_d4j_env() -> Dict[str, str]:
    """Build deterministic execution environment for Defects4J across all 5 projects.

    - TZ: America/New_York (fixes Joda-Time timezone & DST transitions)
    - LC_ALL/LANG: en_US.UTF-8 (or C.UTF-8 when en_US.UTF-8 is not generated, preventing ASCII fallback)
    - ANT_OPTS: -Xmx2048m -Dfile.encoding=UTF-8 (fixes Closure Compiler Ant heap and encodings)
    - JAVA_TOOL_OPTIONS: -Djava.awt.headless=true -Dfile.encoding=UTF-8 (fixes AWT & UTF-8 javac on Math-17..35)
    """
    global _CACHED_UTF8_LOCALE
    if _CACHED_UTF8_LOCALE is None:
        _CACHED_UTF8_LOCALE = _resolve_utf8_locale()
    env = dict(os.environ)
    env["TZ"] = "America/New_York"
    env["LC_ALL"] = _CACHED_UTF8_LOCALE
    env["LANG"] = _CACHED_UTF8_LOCALE
    env.setdefault("ANT_OPTS", "-Xmx2048m -Dfile.encoding=UTF-8")
    java_opts = env.get("JAVA_TOOL_OPTIONS", "-Djava.awt.headless=true")
    if "-Dfile.encoding=" not in java_opts:
        java_opts = f"{java_opts} -Dfile.encoding=UTF-8".strip()
    env["JAVA_TOOL_OPTIONS"] = java_opts
    return env


def wrap_d4j_cmd(cmd: List[str], is_chart: bool = False) -> List[str]:
    """Prefix Defects4J test commands with xvfb-run for Chart in Linux if available."""
    if is_chart and shutil.which("xvfb-run"):
        return ["xvfb-run", "-a", "-s", "-screen 0 1024x768x24", *cmd]
    return cmd


class SandboxSecurityError(PermissionError):
    """Raised when a tool request attempts to escape the pre-fix root or read blocked tests."""


@dataclass
class TestRunFeedback:
    """Three execution-derived feedback signals routed back to the agent (Proposal Sec. 4.2)."""

    compiled: bool
    compiler_output: str
    passed: Optional[bool]
    failure_type: str               # PASS | ASSERTION_FAILURE | RUNTIME_ERROR | COMPILE_ERROR | TIMEOUT
    exception_type: str
    exception_message: str
    stack_trace: str
    enclosing_method_covered_lines: List[int]
    enclosing_method_missed_lines: List[int]
    enclosing_method_hit: bool

    def to_agent_text(self) -> str:
        """Render verbatim, non-gamified diagnostic feedback for the agent."""
        parts = [f"### Compilation Status: {'SUCCESS' if self.compiled else 'FAILED'}"]
        if self.compiler_output.strip():
            parts.append("#### Compiler Output (verbatim):\n```\n" + self.compiler_output.strip() + "\n```")
        if not self.compiled:
            return "\n\n".join(parts)

        outcome_str = "PASS" if self.passed else f"FAIL ({self.failure_type})"
        parts.append(f"### Pre-Fix Execution Outcome: {outcome_str}")
        if not self.passed:
            if self.exception_type:
                parts.append(f"- **Exception / Assertion Type:** `{self.exception_type}`")
            if self.exception_message:
                parts.append(f"- **Message:** {self.exception_message}")
            if self.stack_trace:
                parts.append("#### Stack Trace:\n```\n" + self.stack_trace.strip() + "\n```")

        # Diagnostic-only coverage signal scoped strictly to the enclosing method (Proposal Sec. 4.2)
        parts.append("### Diagnostic Coverage Signal (Enclosing Method on Pre-Fix Version)")
        if self.enclosing_method_hit:
            parts.append(
                f"- Your test **entered** the enclosing method.\n"
                f"- Lines executed inside enclosing method: {self.enclosing_method_covered_lines}\n"
                f"- Lines NOT executed inside enclosing method: {self.enclosing_method_missed_lines}"
            )
        else:
            parts.append(
                "- **Diagnostic note:** Your test did **not** execute any lines of the enclosing method "
                "(it may be exercising the wrong code path or throwing before reaching the target method)."
            )
        return "\n\n".join(parts)


def strip_triggering_test_methods(
    workspace_dir: Path,
    dir_src_tests: str,
    triggering_tests: List[str],
) -> List[str]:
    """Physically redact pre-existing Defects4J triggering test methods from the agent checkout.

    Replaces each triggering test method body with a stub comment so the project still
    compiles cleanly while removing any ground-truth bug-reproducing logic or assertions.
    Returns the list of relative test file paths that contained triggering tests.
    """
    redacted_files: List[str] = []
    by_class: Dict[str, Set[str]] = {}
    for entry in triggering_tests:
        if "::" in entry:
            cls_name, method_name = entry.split("::", 1)
            by_class.setdefault(cls_name.strip(), set()).add(method_name.strip())

    for cls_name, methods in by_class.items():
        rel_path = f"{dir_src_tests}/{cls_name.replace('.', '/')}.java"
        full_path = workspace_dir / rel_path
        if not full_path.exists():
            continue
        redacted_files.append(rel_path)
        text = full_path.read_text(encoding="utf-8", errors="replace")
        for m_name in methods:
            text = _redact_java_method(text, m_name)
        full_path.write_text(text, encoding="utf-8")

    return redacted_files


def find_matching_brace(source: str, start_brace_idx: int) -> int:
    """Find index of the matching closing brace, ignoring braces in strings and comments."""
    depth = 0
    idx = start_brace_idx
    n = len(source)
    in_string = False
    in_char = False
    in_line_comment = False
    in_block_comment = False

    while idx < n:
        ch = source[idx]
        nxt = source[idx + 1] if idx + 1 < n else ""

        if in_line_comment:
            if ch == "\n":
                in_line_comment = False
        elif in_block_comment:
            if ch == "*" and nxt == "/":
                in_block_comment = False
                idx += 1
        elif in_string:
            if ch == "\\":
                idx += 1  # Skip escaped character
            elif ch == '"':
                in_string = False
        elif in_char:
            if ch == "\\":
                idx += 1  # Skip escaped character
            elif ch == "'":
                in_char = False
        else:
            if ch == "/" and nxt == "/":
                in_line_comment = True
                idx += 1
            elif ch == "/" and nxt == "*":
                in_block_comment = True
                idx += 1
            elif ch == '"':
                in_string = True
            elif ch == "'":
                in_char = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return idx
        idx += 1
    return -1


def _redact_java_method(source: str, method_name: str) -> str:
    """Replace the body of `method_name` with an empty placeholder while preserving syntax."""
    pattern = re.compile(
        rf"((?:@\w+(?:\([^)]*\))?\s*)*"
        rf"(?:public|protected|private)?\s*(?:static\s+)?(?:<[\w,\s?]+>\s+)?void\s+{re.escape(method_name)}\s*\([^)]*\)"
        rf"(?:\s*throws\s+[\w.,\s]+)?\s*)\{{",
        re.MULTILINE,
    )
    match = pattern.search(source)
    if not match:
        return source
    start_brace = match.end() - 1
    end_brace = find_matching_brace(source, start_brace)
    if end_brace == -1:
        return source

    replacement = f"{match.group(1)}{{\n        // [REDACTED: pre-existing triggering test isolated from agent]\n    }}"
    return source[: match.start()] + replacement + source[end_brace + 1 :]


def parse_enclosing_method_coverage(
    coverage_xml_path: Path,
    target_class_fqn: str,
    start_line: int,
    end_line: int,
) -> Tuple[List[int], List[int]]:
    """Parse Cobertura `coverage.xml` and return (covered_lines, missed_lines) in `[start_line, end_line]`."""
    if not coverage_xml_path.exists():
        return [], []
    try:
        tree = ET.parse(coverage_xml_path)
    except ET.ParseError:
        return [], []

    covered: List[int] = []
    missed: List[int] = []
    root = tree.getroot()
    for cls_el in root.findall(".//class"):
        name = cls_el.attrib.get("name", "")
        if name == target_class_fqn or name.startswith(f"{target_class_fqn}$"):
            for line_el in cls_el.findall("./lines/line"):
                num = int(line_el.attrib["number"])
                if start_line <= num <= end_line:
                    hits = int(line_el.attrib.get("hits", "0"))
                    if hits > 0:
                        covered.append(num)
                    else:
                        missed.append(num)
    return sorted(set(covered)), sorted(set(missed))


def classify_junit_failure(failing_tests_log: str, test_class_simple_name: str = "") -> Tuple[str, str, str, str]:
    """Classify JUnit failure trace into (failure_type, exception_type, message, stack_trace)."""
    text = failing_tests_log.strip()
    if not text:
        return "RUNTIME_ERROR", "UnknownFailure", "Test failed with empty failing_tests output", ""

    lines = text.splitlines()
    target_idx = 0
    if test_class_simple_name:
        for idx, ln in enumerate(lines):
            if ln.startswith("---") and test_class_simple_name in ln:
                target_idx = idx + 1
                break
    relevant_lines = lines[target_idx:]

    exc_line = ""
    for ln in relevant_lines:
        if ln.startswith("---"):
            # Reached next test failure block
            break
        if ":" in ln or "Error" in ln or "Exception" in ln:
            exc_line = ln.strip()
            break

    exc_type = exc_line.split(":", 1)[0].strip() if exc_line else "UnknownException"
    exc_msg = exc_line.split(":", 1)[1].strip() if ":" in exc_line else ""

    exc_lower = exc_type.lower()
    is_assertion = any(tok in exc_lower for tok in ("assertion", "comparisonfailure"))
    failure_type = "ASSERTION_FAILURE" if is_assertion else "RUNTIME_ERROR"
    return failure_type, exc_type, exc_msg, "\n".join(relevant_lines[:40])


class PrefixSandbox:
    """Isolated pre-fix checkout backing the MCP server for a single Defects4J bug."""

    def __init__(
        self,
        workspace_root: Path,
        dir_src_tests: str,
        target_class_fqn: str,
        enclosing_method_start: int,
        enclosing_method_end: int,
        blocked_rel_paths: Optional[Set[str]] = None,
        d4j_bin: str = "defects4j",
        compile_timeout: int = 120,
        test_timeout: int = 120,
        coverage_timeout: int = 180,
    ) -> None:
        self.root = workspace_root.resolve()
        self.dir_src_tests = dir_src_tests
        self.target_class_fqn = target_class_fqn
        self.enclosing_method_start = enclosing_method_start
        self.enclosing_method_end = enclosing_method_end
        self.blocked_rel_paths = {p.lstrip("/") for p in (blocked_rel_paths or set())}
        self.d4j_bin = d4j_bin
        self.compile_timeout = compile_timeout
        self.test_timeout = test_timeout
        self.coverage_timeout = coverage_timeout

    @classmethod
    def materialize_clean(
        cls,
        raw_prefix_checkout: Path,
        agent_workspace_dir: Path,
        dir_src_tests: str,
        target_class_fqn: str,
        enclosing_method_start: int,
        enclosing_method_end: int,
        triggering_tests: List[str],
        strip_triggers: bool = True,
        **kwargs,
    ) -> "PrefixSandbox":
        """Copy raw pre-fix checkout into an isolated agent workspace and strip triggering tests."""
        if agent_workspace_dir.exists():
            shutil.rmtree(agent_workspace_dir)
        shutil.copytree(raw_prefix_checkout, agent_workspace_dir, symlinks=False)

        blocked: Set[str] = set()
        if strip_triggers:
            redacted = strip_triggering_test_methods(
                agent_workspace_dir, dir_src_tests, triggering_tests
            )
            blocked.update(redacted)

        return cls(
            workspace_root=agent_workspace_dir,
            dir_src_tests=dir_src_tests,
            target_class_fqn=target_class_fqn,
            enclosing_method_start=enclosing_method_start,
            enclosing_method_end=enclosing_method_end,
            blocked_rel_paths=blocked,
            **kwargs,
        )

    def _resolve_safe(self, rel_path: str) -> Path:
        """Validate that `rel_path` stays strictly inside `self.root` and is not denylisted."""
        if not rel_path or Path(rel_path).is_absolute():
            raise SandboxSecurityError(f"Absolute or empty path rejected: {rel_path!r}")
        normalized_rel = Path(rel_path).as_posix().lstrip("./")
        if normalized_rel in self.blocked_rel_paths:
            raise SandboxSecurityError(
                f"Access denied: {rel_path!r} is an isolated pre-existing triggering test."
            )
        candidate = (self.root / rel_path).resolve()
        try:
            candidate.relative_to(self.root)
        except ValueError as exc:
            raise SandboxSecurityError(
                f"Path traversal rejected: {rel_path!r} resolves outside pre-fix sandbox."
            ) from exc
        # Double-check resolved relative path against denylist
        resolved_rel = candidate.relative_to(self.root).as_posix()
        if resolved_rel in self.blocked_rel_paths:
            raise SandboxSecurityError(
                f"Access denied: {resolved_rel!r} is an isolated pre-existing triggering test."
            )
        return candidate

    def get_context(
        self,
        rel_path: str,
        start_line: int = 1,
        end_line: int = 200,
    ) -> str:
        """Read a bounded slice of a source file strictly from the pre-fix checkout."""
        target = self._resolve_safe(rel_path)
        if not target.exists() or not target.is_file():
            return f"File not found in pre-fix workspace: {rel_path}"
        lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
        s = max(1, start_line)
        e = min(len(lines), max(s, end_line))
        numbered = [f"{idx:4d} | {lines[idx - 1]}" for idx in range(s, e + 1)]
        return f"// File: {rel_path} (pre-fix lines {s}-{e} of {len(lines)})\n" + "\n".join(numbered)

    def run_test(
        self,
        package_name: str,
        test_class_name: str,
        test_method_name: str,
        java_source: str,
    ) -> TestRunFeedback:
        """Write candidate JUnit test into pre-fix workspace, compile, run, and measure method coverage."""
        # Clean source if wrapped in markdown code fences
        java_source = java_source.strip()
        if java_source.startswith("```"):
            lines = java_source.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            java_source = "\n".join(lines).strip()

        # If LLM declared a public class name different from test_class_name, align them
        pub_m = re.search(r"\bpublic\s+(?:final\s+)?class\s+([A-Za-z_]\w*)", java_source)
        if pub_m and pub_m.group(1) != test_class_name:
            actual_cls = pub_m.group(1)
            # Replace declared class name with test_class_name to preserve consistent targeting
            java_source = re.sub(
                rf"\bpublic\s+(?:final\s+)?class\s+{re.escape(actual_cls)}\b",
                f"public class {test_class_name}",
                java_source,
                count=1,
            )

        if not re.match(r"^[A-Za-z_][\w]*$", test_class_name):
            raise SandboxSecurityError(f"Invalid Java class name: {test_class_name!r}")
        pkg_path = package_name.replace(".", "/") if package_name else ""
        rel_test_file = (
            f"{self.dir_src_tests}/{pkg_path}/{test_class_name}.java"
            if pkg_path
            else f"{self.dir_src_tests}/{test_class_name}.java"
        )
        test_file = self._resolve_safe(rel_test_file)
        test_file.parent.mkdir(parents=True, exist_ok=True)
        test_file.write_text(java_source, encoding="utf-8")

        is_chart = "org.jfree" in self.target_class_fqn or "Chart" in str(self.root)
        d4j_env = get_d4j_env()

        # 1. Compile on pre-fix
        try:
            comp_cmd = wrap_d4j_cmd([self.d4j_bin, "compile"], is_chart=is_chart)
            comp = subprocess.run(
                comp_cmd,
                cwd=str(self.root),
                capture_output=True,
                text=True,
                errors="replace",
                timeout=self.compile_timeout,
                env=d4j_env,
            )
        except subprocess.TimeoutExpired:
            return TestRunFeedback(
                compiled=False,
                compiler_output=f"Compilation timed out after {self.compile_timeout}s",
                passed=None,
                failure_type="TIMEOUT",
                exception_type="CompileTimeout",
                exception_message="defects4j compile timed out",
                stack_trace="",
                enclosing_method_covered_lines=[],
                enclosing_method_missed_lines=[],
                enclosing_method_hit=False,
            )

        compiler_out = (comp.stdout + "\n" + comp.stderr).strip()
        if comp.returncode != 0:
            return TestRunFeedback(
                compiled=False,
                compiler_output=compiler_out,
                passed=None,
                failure_type="COMPILE_ERROR",
                exception_type="CompileError",
                exception_message="Candidate test failed to compile on pre-fix version",
                stack_trace="",
                enclosing_method_covered_lines=[],
                enclosing_method_missed_lines=[],
                enclosing_method_hit=False,
            )

        fqn_test = f"{package_name}.{test_class_name}::{test_method_name}" if package_name else f"{test_class_name}::{test_method_name}"

        # Clean stale failing_tests log before running
        failing_log_path = self.root / "failing_tests"
        if failing_log_path.exists():
            failing_log_path.unlink()

        # 2. Run test on pre-fix
        try:
            test_cmd = wrap_d4j_cmd([self.d4j_bin, "test", "-t", fqn_test], is_chart=is_chart)
            run_proc = subprocess.run(
                test_cmd,
                cwd=str(self.root),
                capture_output=True,
                text=True,
                errors="replace",
                timeout=self.test_timeout,
                env=d4j_env,
            )
        except subprocess.TimeoutExpired:
            return TestRunFeedback(
                compiled=True,
                compiler_output=compiler_out,
                passed=False,
                failure_type="TIMEOUT",
                exception_type="TestTimeout",
                exception_message=f"defects4j test timed out after {self.test_timeout}s",
                stack_trace="",
                enclosing_method_covered_lines=[],
                enclosing_method_missed_lines=[],
                enclosing_method_hit=False,
            )

        passed = run_proc.returncode == 0 and "Failing tests: 0" in run_proc.stdout
        if passed:
            failure_type, exc_type, exc_msg, trace = "PASS", "", "", ""
        else:
            failing_log = (
                failing_log_path.read_text(encoding="utf-8", errors="replace")
                if failing_log_path.exists()
                else (run_proc.stdout + "\n" + run_proc.stderr)
            )
            failure_type, exc_type, exc_msg, trace = classify_junit_failure(
                failing_log, test_class_name
            )

        # 3. Diagnostic coverage scoped strictly to enclosing method on pre-fix
        cov_xml_path = self.root / "coverage.xml"
        if cov_xml_path.exists():
            cov_xml_path.unlink()

        covered, missed = [], []
        try:
            cov_cmd = wrap_d4j_cmd([self.d4j_bin, "coverage", "-t", fqn_test], is_chart=is_chart)
            subprocess.run(
                cov_cmd,
                cwd=str(self.root),
                capture_output=True,
                text=True,
                errors="replace",
                timeout=self.coverage_timeout,
                env=d4j_env,
            )
            covered, missed = parse_enclosing_method_coverage(
                self.root / "coverage.xml",
                self.target_class_fqn,
                self.enclosing_method_start,
                self.enclosing_method_end,
            )
        except subprocess.TimeoutExpired:
            pass

        return TestRunFeedback(
            compiled=True,
            compiler_output=compiler_out,
            passed=passed,
            failure_type=failure_type,
            exception_type=exc_type,
            exception_message=exc_msg,
            stack_trace=trace,
            enclosing_method_covered_lines=covered,
            enclosing_method_missed_lines=missed,
            enclosing_method_hit=len(covered) > 0,
        )
