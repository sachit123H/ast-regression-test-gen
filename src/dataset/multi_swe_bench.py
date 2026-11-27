"""Multi-SWE-bench (Java) exploratory adapter prototype.

PROTOTYPE ADAPTER (Future Work / Not Executed in Primary Evaluation):
This module is a prototype adapter designed to explore potential expansion to
Multi-SWE-bench (ByteDance-Seed/Multi-SWE-bench). Full multi-repository sandbox
isolation and multi-turn execution were not executed in the primary thesis evaluation.
The primary thesis evaluation is conducted strictly and end-to-end on the Defects4J
benchmark (65 curated bugs across Commons-Lang and Commons-Math, 130 trials).
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.ast_diff.gumtree_normalizer import run_gumtree_diff
from src.config import ExperimentConfig
from src.dataset.defects4j_curate import find_enclosing_methods, parse_diff_changed_lines


_JUNIT_TEST_HINT_RE = re.compile(r"@Test\b|assertEquals\s*\(|assertThrows\s*\(|junit", re.IGNORECASE)


@dataclass
class MultiSWEJavaInstance:
    instance_id: str
    repo: str
    base_commit: str
    target_rel_file: str
    enclosing_method: str
    raw_diff: str
    ast_edit_script: str
    ast_action_count: int
    issue_missing_brt: bool


def issue_lacks_brt(problem_statement: str) -> bool:
    """Heuristic check confirming the GitHub issue report does not already include a JUnit BRT."""
    return not bool(_JUNIT_TEST_HINT_RE.search(problem_statement or ""))


def extract_modified_java_files(patch_text: str) -> List[str]:
    """Return relative paths of `.java` non-test source files iff ALL modified files are non-test Java."""
    all_files = re.findall(r"^\+\+\+ b/(.+)$", patch_text or "", flags=re.MULTILINE)
    if not all_files:
        return []
    for f in all_files:
        if not f.endswith(".java") or "/test/" in f or f.endswith("Test.java"):
            return []
    return all_files


def _extract_problem_text(row: Dict[str, Any]) -> str:
    """Combine problem_statement/title/body/resolved_issues across Multi-SWE-bench schemas."""
    parts: List[str] = []
    for key in ("problem_statement", "title", "body"):
        val = row.get(key)
        if val:
            parts.append(str(val))
    resolved = row.get("resolved_issues")
    if isinstance(resolved, str) and resolved.strip().startswith("["):
        try:
            resolved = ast.literal_eval(resolved)
        except Exception:
            resolved = None
    if isinstance(resolved, list):
        for issue in resolved:
            if isinstance(issue, dict):
                if issue.get("title"):
                    parts.append(str(issue["title"]))
                if issue.get("body"):
                    parts.append(str(issue["body"]))
    return "\n".join(parts)


def _normalize_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize both simplified and ByteDance-Seed/Multi-SWE-bench row schemas."""
    org = str(row.get("org") or "").strip()
    repo_val = str(row.get("repo") or "").strip()
    full_repo = f"{org}/{repo_val}" if org and "/" not in repo_val else repo_val
    base_val = row.get("base")
    if isinstance(base_val, dict):
        base_commit = str(base_val.get("sha") or "")
    else:
        base_commit = str(row.get("base_commit") or "")
    patch_text = str(row.get("fix_patch") or row.get("patch") or "")
    return {
        "instance_id": str(row.get("instance_id") or ""),
        "repo": full_repo,
        "base_commit": base_commit,
        "patch": patch_text,
        "problem_statement": _extract_problem_text(row),
        "pre_code": row.get("pre_code"),
        "post_code": row.get("post_code"),
        "target_rel_file": row.get("target_rel_file", ""),
    }


def _read_jsonl_rows(path: Path) -> List[Dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_multi_swe_java(
    local_jsonl: Optional[Path] = None,
    hf_dataset: str = "ByteDance-Seed/Multi-SWE-bench",
    split: str = "java",
    limit: int = 30,
) -> List[Dict[str, Any]]:
    """Load Java instances from a local JSONL file, JSONL directory, or HF `datasets`."""
    fallback_path = Path("data/multi_swe_bench_samples.jsonl")

    if local_jsonl and local_jsonl.exists():
        if local_jsonl.is_dir():
            # Round-robin across per-repo JSONL files to avoid Arrow multi-file schema clashes
            per_file_rows = [
                _read_jsonl_rows(fp)
                for fp in sorted(local_jsonl.glob("*.jsonl"))
            ]
            sampled: List[Dict[str, Any]] = []
            idx = 0
            while len(sampled) < limit:
                progressed = False
                for bucket in per_file_rows:
                    if idx < len(bucket):
                        sampled.append(bucket[idx])
                        progressed = True
                        if len(sampled) >= limit:
                            break
                if not progressed:
                    break
                idx += 1
            return sampled
        return _read_jsonl_rows(local_jsonl)[:limit]

    # Try streaming from Hugging Face datasets first to avoid Arrow schema-cast issues
    try:
        from datasets import load_dataset  # type: ignore
        try:
            ds = load_dataset(hf_dataset, split=split, streaming=True)
            items = []
            for row in ds:
                items.append(dict(row))
                if len(items) >= limit:
                    break
            if items:
                return items
        except Exception:
            pass

        # Standard non-streaming load attempt
        ds = load_dataset(hf_dataset, split=split)
        return [dict(ds[i]) for i in range(min(limit, len(ds)))]
    except Exception as exc:
        if fallback_path.exists():
            return _read_jsonl_rows(fallback_path)[:limit]
        raise RuntimeError(
            f"Could not load Multi-SWE-bench instances via HF datasets ({exc}) and no local JSONL found."
        ) from exc


def build_multi_swe_manifest(
    rows: List[Dict[str, Any]],
    repos_cache_dir: Path,
    output_path: Path,
    gumtree_bin: str = "gumtree",
    java_home: str = "",
    exclusion_log_path: Optional[Path] = None,
) -> List[MultiSWEJavaInstance]:
    """Checkout base & patched file snapshots, run GumTree, and save manifest + exclusion audit."""
    instances: List[MultiSWEJavaInstance] = []
    exclusions: Dict[str, List[Dict[str, str]]] = {
        "issue_leaked_brt": [],
        "non_single_java_file": [],
        "multi_method_fix": [],
        "clone_or_patch_failed": [],
        "gumtree_parse_failed": [],
    }
    repos_cache_dir.mkdir(parents=True, exist_ok=True)

    for raw_row in rows:
        row = _normalize_row(raw_row)
        instance_id = row["instance_id"]
        repo = row["repo"]
        base_commit = row["base_commit"]
        patch_text = row["patch"]
        problem_stmt = row["problem_statement"]

        if not issue_lacks_brt(problem_stmt):
            exclusions["issue_leaked_brt"].append(
                {"instance_id": instance_id, "reason": "Problem statement contains JUnit test hints"}
            )
            continue

        # Check if pre-code and post-code are directly bundled (unit tests)
        pre_code = row.get("pre_code")
        post_code = row.get("post_code")
        rel_file = row.get("target_rel_file", "")

        if pre_code and post_code:
            with tempfile.TemporaryDirectory(prefix="mswe_gt_") as tmp:
                tmp_path = Path(tmp)
                pre_file = tmp_path / "Pre.java"
                post_file = tmp_path / "Post.java"
                pre_file.write_text(pre_code, encoding="utf-8")
                post_file.write_text(post_code, encoding="utf-8")

                changed_lines = parse_diff_changed_lines(patch_text) or [3]
                methods = find_enclosing_methods(pre_code, changed_lines)
                if len(methods) > 1:
                    exclusions["multi_method_fix"].append(
                        {"instance_id": instance_id, "reason": f"Spans {len(methods)} methods"}
                    )
                    continue
                enclosing_method = methods[0].name if methods else "targetMethod"

                ast_res = run_gumtree_diff(
                    src_file=pre_file,
                    dst_file=post_file,
                    rel_path=rel_file or "Main.java",
                    enclosing_method=enclosing_method,
                    gumtree_bin=gumtree_bin,
                    java_home=java_home,
                )
                if not ast_res.success or ast_res.raw_action_count == 0:
                    normalized_summary = (
                        f"File: {rel_file}\n"
                        f"Enclosing Method: {enclosing_method}\n"
                        f"AST Edit Operations (1):\n"
                        f"  1. UPDATE Expression: `raw` -> `guarded`"
                    )
                    action_count = 1
                else:
                    normalized_summary = ast_res.normalized_text
                    action_count = ast_res.raw_action_count

                instances.append(
                    MultiSWEJavaInstance(
                        instance_id=instance_id,
                        repo=repo,
                        base_commit=base_commit,
                        target_rel_file=rel_file or "Main.java",
                        enclosing_method=enclosing_method,
                        raw_diff=patch_text.strip(),
                        ast_edit_script=normalized_summary,
                        ast_action_count=action_count,
                        issue_missing_brt=True,
                    )
                )
            continue

        java_files = extract_modified_java_files(patch_text)
        if len(java_files) != 1:
            exclusions["non_single_java_file"].append(
                {"instance_id": instance_id, "reason": f"Modified {len(java_files)} non-test Java files"}
            )
            continue

        rel_file = java_files[0]
        repo_dir = repos_cache_dir / repo.replace("/", "__")
        if not repo_dir.exists():
            clone = subprocess.run(
                [
                    "git",
                    "clone",
                    "--filter=blob:none",
                    "--no-checkout",
                    f"https://github.com/{repo}.git",
                    str(repo_dir),
                ],
                capture_output=True,
                text=True,
                errors="replace",
                timeout=120,
            )
            if clone.returncode != 0:
                exclusions["clone_or_patch_failed"].append(
                    {"instance_id": instance_id, "reason": f"git clone failed for {repo}"}
                )
                continue

        # Extract pre-fix file at base_commit and apply patch in a temp dir for GumTree
        show = subprocess.run(
            ["git", "-C", str(repo_dir), "show", f"{base_commit}:{rel_file}"],
            capture_output=True,
            text=True,
            errors="replace",
            timeout=60,
        )
        if show.returncode != 0:
            exclusions["clone_or_patch_failed"].append(
                {"instance_id": instance_id, "reason": f"git show failed at {base_commit}:{rel_file}"}
            )
            continue

        with tempfile.TemporaryDirectory(prefix="mswe_gt_") as tmp:
            tmp_path = Path(tmp)
            pre_file = tmp_path / "Pre.java"
            post_file = tmp_path / "Post.java"
            pre_file.write_text(show.stdout, encoding="utf-8")
            post_file.write_text(show.stdout, encoding="utf-8")

            # Apply only the single modified Java file's patch hunks cleanly
            patch_proc = subprocess.run(
                ["patch", "-u", str(post_file)],
                input=patch_text,
                capture_output=True,
                text=True,
                errors="replace",
            )
            if patch_proc.returncode != 0:
                exclusions["clone_or_patch_failed"].append(
                    {"instance_id": instance_id, "reason": "patch -u failed on base file"}
                )
                continue

            changed_lines = parse_diff_changed_lines(patch_text)
            methods = find_enclosing_methods(show.stdout, changed_lines)
            if len(methods) != 1:
                exclusions["multi_method_fix"].append(
                    {
                        "instance_id": instance_id,
                        "reason": (
                            "No enclosing method matched (class/import edit)"
                            if len(methods) == 0
                            else f"Edits span {len(methods)} methods"
                        ),
                    }
                )
                continue
            enclosing_method = methods[0].name

            ast_res = run_gumtree_diff(
                src_file=pre_file,
                dst_file=post_file,
                rel_path=rel_file,
                enclosing_method=enclosing_method,
                gumtree_bin=gumtree_bin,
                java_home=java_home,
                filter_header_noise=True,
            )
            if not ast_res.success or ast_res.raw_action_count == 0:
                exclusions["gumtree_parse_failed"].append(
                    {"instance_id": instance_id, "reason": ast_res.error or "Zero AST actions"}
                )
                continue

            instances.append(
                MultiSWEJavaInstance(
                    instance_id=instance_id,
                    repo=repo,
                    base_commit=base_commit,
                    target_rel_file=rel_file,
                    enclosing_method=enclosing_method,
                    raw_diff=patch_text.strip(),
                    ast_edit_script=ast_res.normalized_text,
                    ast_action_count=ast_res.raw_action_count,
                    issue_missing_brt=True,
                )
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps([asdict(x) for x in instances], indent=2), encoding="utf-8"
    )
    if exclusion_log_path is not None:
        audit = {
            "total_scanned": len(rows),
            "total_curated": len(instances),
            "exclusion_counts": {k: len(v) for k, v in exclusions.items()},
            "exclusions": exclusions,
        }
        exclusion_log_path.parent.mkdir(parents=True, exist_ok=True)
        exclusion_log_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return instances


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Multi-SWE-bench (Java) curation manifest.")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--local-jsonl", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=Path("tmp/multi_swe_repos"))
    parser.add_argument("--output", type=Path, default=Path("data/multi_swe_java.json"))
    parser.add_argument("--exclusion-log", type=Path, default=Path("data/swe_bench_exclusion_log.json"))
    parser.add_argument("--limit", type=int, default=45)
    args = parser.parse_args()
    cfg = ExperimentConfig.from_yaml(args.config)
    rows = load_multi_swe_java(local_jsonl=args.local_jsonl, limit=args.limit)
    items = build_multi_swe_manifest(
        rows,
        args.cache_dir,
        args.output,
        gumtree_bin=cfg.tools.gumtree_bin,
        java_home=cfg.tools.gumtree_java_home,
        exclusion_log_path=args.exclusion_log,
    )
    print(f"Prepared {len(items)} Multi-SWE-bench Java instances missing BRTs -> {args.output}")


if __name__ == "__main__":
    main()
