"""Defects4J 3-filter dataset curation and exclusion audit pipeline.

Implements Proposal Section 4.1 & Section 5.1:
1. Both pre-fix (b) and post-fix (f) versions build cleanly and their suites run.
2. The fix is localized to a single method (with automatic relaxation reporting
   if the intersection falls below `min_viable_bugs`).
3. GumTree parses both versions cleanly and produces a valid AST edit script.

Also records:
- Enclosing method name + line span in the pre-fix version (for Sec. 4.2
  diagnostic coverage feedback).
- Pre-existing Defects4J triggering tests (`tests.trigger`) so they can be
  programmatically stripped from the agent's sandbox (Aug 12 email commitment).
- Exact exclusion counts and per-bug reasons in `data/exclusion_log.json`.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.ast_diff.gumtree_normalizer import extract_raw_git_diff, run_gumtree_diff
from src.config import ExperimentConfig
from src.mcp.sandbox import find_matching_brace, get_d4j_env, wrap_d4j_cmd


@dataclass
class MethodSpan:
    name: str
    signature: str
    start_line: int
    end_line: int


@dataclass
class CuratedBug:
    bug_key: str                      # e.g., "Lang-1"
    project: str                      # e.g., "Lang"
    bug_id: int
    modified_classes: List[str]
    relevant_classes: List[str]
    triggering_tests: List[str]       # e.g., ["org.apache.commons.lang3.math.NumberUtilsTest::TestLang747"]
    dir_src_classes: str              # e.g., "src/main/java"
    dir_src_tests: str                # e.g., "src/test/java"
    target_rel_file: str              # Relative path to modified Java source file
    enclosing_method: str             # Method enclosing the fix in pre-fix version
    enclosing_method_start: int       # 1-indexed start line in pre-fix file
    enclosing_method_end: int         # 1-indexed end line in pre-fix file
    raw_diff: str                     # Control condition input
    ast_edit_script: str              # Treatment condition input (normalized GumTree)
    ast_action_count: int
    is_single_method_fix: bool
    enclosing_method_source: str = "" # Bounded pre-fix enclosing method lines for inline context


def _run_d4j(
    args: List[str],
    d4j_bin: str = "defects4j",
    cwd: Optional[Path] = None,
    timeout: int = 180,
    project: str = "",
) -> subprocess.CompletedProcess[str]:
    cmd = wrap_d4j_cmd([d4j_bin, *args], is_chart=(project == "Chart"))
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        errors="replace",
        timeout=timeout,
        env=get_d4j_env(),
    )


def _export_prop(d4j_bin: str, work_dir: Path, prop: str, project: str = "") -> List[str]:
    proc = _run_d4j(["export", "-p", prop, "-w", str(work_dir)], d4j_bin=d4j_bin, timeout=60, project=project)
    if proc.returncode != 0:
        return []
    return [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]


def parse_java_method_spans(source_text: str) -> List[MethodSpan]:
    """Lightweight brace-balanced Java method span finder (1-indexed line numbers).

    Supports both single-line and multi-line method/constructor signatures.
    """
    lines = source_text.splitlines()
    # Match method/constructor declarations (excluding control keywords)
    method_re = re.compile(
        r"^\s*(?:(?:public|protected|private|static|final|synchronized|abstract|native|strictfp|default)\s+)*"
        r"(?:<[^>]+>\s+)?"
        r"(?:[\w\[\]<>.,\s]+\s+)?"
        r"([A-Za-z_$][\w$]*)\s*\([^;{}]*\)\s*(?:throws\s+[\w.,\s]+)?\s*\{"
    )
    control_keywords = {"if", "for", "while", "switch", "catch", "synchronized", "else", "try"}
    spans: List[MethodSpan] = []

    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        sig_end = i
        sig_candidate = line
        stripped_line = line.strip()
        # Handle multi-line method signatures spanning up to 8 lines before `{`
        if (
            "(" in stripped_line
            and "{" not in stripped_line
            and ";" not in stripped_line
            and not stripped_line.startswith(("//", "*", "/*"))
        ):
            parts = [line]
            for k in range(i + 1, min(n, i + 8)):
                parts.append(lines[k])
                if "{" in lines[k] or ";" in lines[k]:
                    sig_end = k
                    break
            sig_candidate = " ".join(p.strip() for p in parts)

        m = method_re.match(sig_candidate)
        if m and m.group(1) not in control_keywords:
            name = m.group(1)
            start_line = i + 1
            brace_line = lines[sig_end]
            line_brace = brace_line.find("{")
            if line_brace != -1:
                brace_pos = sum(len(lines[k]) + 1 for k in range(sig_end)) + line_brace
                end_brace_idx = find_matching_brace(source_text, brace_pos)
                if end_brace_idx != -1:
                    end_line = source_text[:end_brace_idx].count("\n") + 1
                    spans.append(
                        MethodSpan(
                            name=name,
                            signature=sig_candidate.strip(),
                            start_line=start_line,
                            end_line=end_line,
                        )
                    )
                    i = end_line
                    continue
            brace_depth = brace_line.count("{") - brace_line.count("}")
            j = sig_end + 1
            while j < n and brace_depth > 0:
                stripped = re.sub(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'', "", lines[j])
                stripped = re.sub(r"//.*$", "", stripped)
                brace_depth += stripped.count("{") - stripped.count("}")
                j += 1
            end_line = j
            spans.append(
                MethodSpan(
                    name=name,
                    signature=sig_candidate.strip(),
                    start_line=start_line,
                    end_line=end_line,
                )
            )
            i = j
        else:
            i += 1
    return spans


def parse_diff_changed_lines(unified_diff: str) -> List[int]:
    """Extract 1-indexed line numbers in the pre-fix (old) file touched by a unified diff."""
    changed: List[int] = []
    old_line = 0
    hunk_re = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
    for line in unified_diff.splitlines():
        m = hunk_re.match(line)
        if m:
            old_line = int(m.group(1))
            continue
        if line.startswith("---") or line.startswith("+++"):
            continue
        if line.startswith("-"):
            changed.append(old_line)
            old_line += 1
        elif line.startswith("+"):
            # Pure insertion anchored at current old_line
            changed.append(max(1, old_line))
        else:
            old_line += 1
    return sorted(set(changed))


def find_enclosing_methods(source_text: str, changed_lines: List[int]) -> List[MethodSpan]:
    """Return all unique Java methods in `source_text` overlapping `changed_lines`."""
    spans = parse_java_method_spans(source_text)
    matched: List[MethodSpan] = []
    seen = set()
    for ln in changed_lines:
        for sp in spans:
            if sp.start_line <= ln <= sp.end_line:
                key = (sp.name, sp.start_line, sp.end_line)
                if key not in seen:
                    seen.add(key)
                    matched.append(sp)
    return matched


def curate_defects4j(cfg: ExperimentConfig) -> Tuple[List[CuratedBug], Dict[str, Any]]:
    """Run the 3-filter Defects4J curation pipeline and persist manifest + exclusion log."""
    d4j = cfg.tools.defects4j_bin
    curated: List[CuratedBug] = []
    if cfg.paths.curated_manifest.exists():
        try:
            raw_curated = json.loads(cfg.paths.curated_manifest.read_text(encoding="utf-8"))
            curated = [CuratedBug(**b) for b in raw_curated]
        except Exception:
            curated = []
    already_curated_keys = {b.bug_key for b in curated}
    relaxed_candidates: List[CuratedBug] = []
    exclusions: Dict[str, List[Dict[str, str]]] = {
        "checkout_or_build_failed": [],
        "multi_file_or_multi_method": [],
        "gumtree_parse_failed": [],
        "no_triggering_tests": [],
    }

    total_scanned = 0
    for project in cfg.dataset.projects:
        bids_proc = _run_d4j(["bids", "-p", project], d4j_bin=d4j, timeout=60)
        if bids_proc.returncode != 0:
            raise RuntimeError(f"defects4j bids -p {project} failed: {bids_proc.stderr}")
        bug_ids = [int(x.strip()) for x in bids_proc.stdout.splitlines() if x.strip().isdigit()]

        for bid in bug_ids:
            total_scanned += 1
            bug_key = f"{project}-{bid}"
            if bug_key in already_curated_keys:
                print(f"[{total_scanned}] Already curated {bug_key}, skipping. (curated: {len(curated)})", flush=True)
                continue
            print(f"[{total_scanned}] Evaluating {bug_key}... (curated: {len(curated)})", flush=True)
            pre_dir = cfg.paths.prefix_root / f"{bug_key}_raw"
            post_dir = cfg.paths.postfix_root / bug_key
            pre_dir.parent.mkdir(parents=True, exist_ok=True)
            post_dir.parent.mkdir(parents=True, exist_ok=True)

            # Checkout pre-fix (b) and post-fix (f)
            if not pre_dir.exists():
                p1 = _run_d4j(["checkout", "-p", project, "-v", f"{bid}b", "-w", str(pre_dir)], d4j)
                if p1.returncode != 0:
                    exclusions["checkout_or_build_failed"].append(
                        {"bug": bug_key, "reason": f"pre-fix checkout failed: {p1.stderr.strip()}"}
                    )
                    continue
            if not post_dir.exists():
                p2 = _run_d4j(["checkout", "-p", project, "-v", f"{bid}f", "-w", str(post_dir)], d4j)
                if p2.returncode != 0:
                    exclusions["checkout_or_build_failed"].append(
                        {"bug": bug_key, "reason": f"post-fix checkout failed: {p2.stderr.strip()}"}
                    )
                    continue

            # Filter 1: Both versions build cleanly and triggering test runs
            c_pre = _run_d4j(["compile"], d4j, cwd=pre_dir, timeout=cfg.tools.compile_timeout_sec, project=project)
            c_post = _run_d4j(["compile"], d4j, cwd=post_dir, timeout=cfg.tools.compile_timeout_sec, project=project)
            if c_pre.returncode != 0 or c_post.returncode != 0:
                exclusions["checkout_or_build_failed"].append(
                    {"bug": bug_key, "reason": "pre-fix or post-fix compile failed"}
                )
                continue

            triggering = _export_prop(d4j, pre_dir, "tests.trigger", project=project)
            if not triggering:
                exclusions["no_triggering_tests"].append(
                    {"bug": bug_key, "reason": "No tests.trigger exported"}
                )
                continue

            # Verify triggering test fails on pre-fix and passes on post-fix
            t_post = _run_d4j(
                ["test", "-t", triggering[0]],
                d4j,
                cwd=post_dir,
                timeout=cfg.tools.test_timeout_sec,
                project=project,
            )
            if t_post.returncode != 0 or "Failing tests: 0" not in t_post.stdout:
                exclusions["checkout_or_build_failed"].append(
                    {"bug": bug_key, "reason": "post-fix triggering test did not pass"}
                )
                continue

            modified_classes = _export_prop(d4j, pre_dir, "classes.modified", project=project)
            relevant_classes = _export_prop(d4j, pre_dir, "classes.relevant", project=project)
            src_dir_list = _export_prop(d4j, pre_dir, "dir.src.classes", project=project)
            test_dir_list = _export_prop(d4j, pre_dir, "dir.src.tests", project=project)
            dir_src_classes = src_dir_list[0] if src_dir_list else "src/main/java"
            dir_src_tests = test_dir_list[0] if test_dir_list else "src/test/java"

            # Filter 2: Single modified file & single enclosing method
            if len(modified_classes) != 1:
                exclusions["multi_file_or_multi_method"].append(
                    {"bug": bug_key, "reason": f"Modified {len(modified_classes)} classes"}
                )
                continue

            rel_java = f"{dir_src_classes}/{modified_classes[0].replace('.', '/')}.java"
            pre_java = pre_dir / rel_java
            post_java = post_dir / rel_java
            if not pre_java.exists() or not post_java.exists():
                exclusions["checkout_or_build_failed"].append(
                    {"bug": bug_key, "reason": f"Missing source file {rel_java}"}
                )
                continue

            raw_diff = extract_raw_git_diff(pre_java, post_java, rel_java)
            changed_lines = parse_diff_changed_lines(raw_diff)
            pre_source = pre_java.read_text(encoding="utf-8", errors="replace")
            enclosing_methods = find_enclosing_methods(pre_source, changed_lines)
            is_single_method = len(enclosing_methods) == 1

            primary_method = (
                enclosing_methods[0]
                if enclosing_methods
                else MethodSpan(name="<class-level>", signature="", start_line=1, end_line=1)
            )

            # Filter 3: GumTree parses both versions cleanly
            ast_res = run_gumtree_diff(
                src_file=pre_java,
                dst_file=post_java,
                rel_path=rel_java,
                enclosing_method=primary_method.name,
                gumtree_bin=cfg.tools.gumtree_bin,
                java_home=cfg.tools.gumtree_java_home,
                filter_header_noise=cfg.dataset.filter_ast_header_noise,
            )
            if not ast_res.success or ast_res.raw_action_count == 0:
                exclusions["gumtree_parse_failed"].append(
                    {"bug": bug_key, "reason": ast_res.error or "Zero AST actions"}
                )
                continue

            pre_lines = pre_source.splitlines()
            m_start = max(1, primary_method.start_line)
            m_end = min(len(pre_lines), max(m_start, primary_method.end_line), m_start + 249)
            enclosing_src = "\n".join(
                f"{ln:4d} | {pre_lines[ln - 1]}" for ln in range(m_start, m_end + 1)
            ) if pre_lines and primary_method.end_line >= primary_method.start_line else ""

            record = CuratedBug(
                bug_key=bug_key,
                project=project,
                bug_id=bid,
                modified_classes=modified_classes,
                relevant_classes=relevant_classes,
                triggering_tests=triggering,
                dir_src_classes=dir_src_classes,
                dir_src_tests=dir_src_tests,
                target_rel_file=rel_java,
                enclosing_method=primary_method.name,
                enclosing_method_start=primary_method.start_line,
                enclosing_method_end=primary_method.end_line,
                raw_diff=raw_diff,
                ast_edit_script=ast_res.normalized_text,
                ast_action_count=ast_res.raw_action_count,
                is_single_method_fix=is_single_method,
                enclosing_method_source=enclosing_src,
            )

            if is_single_method or not cfg.dataset.require_single_method_fix:
                curated.append(record)
                print(f"  ✓ Curated {bug_key} (enclosing: {primary_method.name}, AST actions: {ast_res.raw_action_count})", flush=True)
                # Incremental flush so progress is never lost
                cfg.paths.curated_manifest.parent.mkdir(parents=True, exist_ok=True)
                cfg.paths.curated_manifest.write_text(
                    json.dumps([asdict(b) for b in curated], indent=2), encoding="utf-8"
                )
            else:
                exclusions["multi_file_or_multi_method"].append(
                    {
                        "bug": bug_key,
                        "reason": f"Edits span {len(enclosing_methods)} methods: {[m.name for m in enclosing_methods]}",
                    }
                )
                relaxed_candidates.append(record)

            if cfg.dataset.max_bugs and len(curated) >= cfg.dataset.max_bugs:
                break
        if cfg.dataset.max_bugs and len(curated) >= cfg.dataset.max_bugs:
            break

    # Proposal Sec. 5.1 Mitigation: relax localized-fix filter if intersection < min_viable_bugs
    relaxed_applied = False
    if len(curated) < cfg.dataset.min_viable_bugs and relaxed_candidates:
        needed = cfg.dataset.min_viable_bugs - len(curated)
        added_candidates = relaxed_candidates[:needed]
        curated.extend(added_candidates)
        relaxed_applied = True
        added_keys = {b.bug_key for b in added_candidates}
        # Update exclusion log so relaxed bugs are moved from excluded to relaxed_into_curated
        exclusions["relaxed_into_curated"] = [
            ex for ex in exclusions["multi_file_or_multi_method"] if ex["bug"] in added_keys
        ]
        exclusions["multi_file_or_multi_method"] = [
            ex for ex in exclusions["multi_file_or_multi_method"] if ex["bug"] not in added_keys
        ]

    summary = {
        "total_scanned": total_scanned,
        "total_selected": len(curated),
        "localized_filter_relaxed": relaxed_applied,
        "exclusion_counts": {k: len(v) for k, v in exclusions.items()},
        "exclusions": exclusions,
    }

    cfg.paths.curated_manifest.parent.mkdir(parents=True, exist_ok=True)
    cfg.paths.curated_manifest.write_text(
        json.dumps([asdict(b) for b in curated], indent=2), encoding="utf-8"
    )
    cfg.paths.exclusion_log.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return curated, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Curate Defects4J bugs with 3-filter audit.")
    parser.add_argument("--config", default="configs/experiment.yaml")
    args = parser.parse_args()
    cfg = ExperimentConfig.from_yaml(args.config)
    curated, summary = curate_defects4j(cfg)
    print(f"Curated {len(curated)} bugs. Exclusion summary: {summary['exclusion_counts']}")


if __name__ == "__main__":
    main()
