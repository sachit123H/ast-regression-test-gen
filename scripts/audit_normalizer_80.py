"""Verify patched GumTree normalizer across all 80 curated Defects4J bugs (65 Lang+Math + 15 aborted Math)."""
from __future__ import annotations

import json
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ast_diff.gumtree_normalizer import run_gumtree_diff, extract_raw_git_diff
from src.config import ExperimentConfig
from src.dataset.defects4j_curate import find_enclosing_methods, parse_diff_changed_lines, _export_prop, _run_d4j


def _audit_one(bk: str, cfg: ExperimentConfig, base_tmp: Path) -> dict:
    proj, bid_str = bk.split("-")
    bid = int(bid_str)
    pre_dir = base_tmp / f"{bk}_b"
    post_dir = base_tmp / f"{bk}_f"
    try:
        _run_d4j(["checkout", "-p", proj, "-v", f"{bid}b", "-w", str(pre_dir)], cfg.tools.defects4j_bin, project=proj)
        _run_d4j(["checkout", "-p", proj, "-v", f"{bid}f", "-w", str(post_dir)], cfg.tools.defects4j_bin, project=proj)

        mod_classes = _export_prop(cfg.tools.defects4j_bin, pre_dir, "classes.modified", project=proj)
        src_dirs = _export_prop(cfg.tools.defects4j_bin, pre_dir, "dir.src.classes", project=proj)
        dir_src = src_dirs[0] if src_dirs else "src/main/java"
        rel_java = f"{dir_src}/{mod_classes[0].replace('.', '/')}.java"
        pre_java = pre_dir / rel_java
        post_java = post_dir / rel_java

        raw_diff = extract_raw_git_diff(pre_java, post_java, rel_java)
        pre_source = pre_java.read_text(encoding="utf-8", errors="replace")
        changed_lines = parse_diff_changed_lines(raw_diff)
        methods = find_enclosing_methods(pre_source, changed_lines)
        method_name = methods[0].name if methods else "<unknown>"

        res = run_gumtree_diff(
            src_file=pre_java,
            dst_file=post_java,
            rel_path=rel_java,
            enclosing_method=method_name,
            gumtree_bin=cfg.tools.gumtree_bin,
            java_home=cfg.tools.gumtree_java_home,
            filter_header_noise=True,
        )

        # Non-leaf nodes lacking a label must carry a recovered subtree_summary;
        # leaf nodes (SimpleName, NumberLiteral, operators) already display their token in label.
        missing_sub = [
            a.format_line()
            for a in res.actions
            if a.action in {"INSERT", "DELETE", "MOVE"} and (not a.label) and (not a.subtree_summary)
        ]
        # Also inspect UPDATE actions with empty label or new_label
        bad_updates = [
            a.format_line()
            for a in res.actions
            if a.action == "UPDATE" and (not a.label and not a.new_label)
        ]
        has_header = [
            a.format_line()
            for a in res.actions
            if a.parent_type in {"TypeDeclaration", "MethodDeclaration"}
            and a.node_type in {"Modifier", "TYPE_DECLARATION_KIND", "SimpleName", "PrimitiveType"}
            and any(
                d.action == "DELETE" and d.node_type == a.node_type and d.label == a.label
                for d in res.actions
            )
        ]
        return {
            "bug_key": bk,
            "success": res.success,
            "error": res.error,
            "action_count": res.raw_action_count,
            "empty_subtree_actions": missing_sub,
            "empty_update_actions": bad_updates,
            "header_noise_actions": has_header,
            "normalized_text": res.normalized_text,
        }
    finally:
        shutil.rmtree(pre_dir, ignore_errors=True)
        shutil.rmtree(post_dir, ignore_errors=True)


def main() -> None:
    cfg = ExperimentConfig.from_yaml("configs/experiment.yaml")
    curated_65 = json.loads(Path("data/curated_bugs.json").read_text(encoding="utf-8"))
    excl = json.loads(Path("data/exclusion_log.json").read_text(encoding="utf-8"))

    bug_keys = [b["bug_key"] for b in curated_65] + [
        item["bug_key"] for item in excl.get("api_quota_exhausted_aborted_bugs", [])
    ]
    print(f"Auditing patched normalizer on {len(bug_keys)} bugs with 6 workers...", flush=True)

    results_map = {}
    with tempfile.TemporaryDirectory(prefix="gt_audit_") as tmp:
        tmp_dir = Path(tmp)
        with ThreadPoolExecutor(max_workers=6) as pool:
            futs = {pool.submit(_audit_one, bk, cfg, tmp_dir): bk for bk in bug_keys}
            for done_idx, fut in enumerate(as_completed(futs), start=1):
                rec = fut.result()
                bk = rec["bug_key"]
                results_map[bk] = rec
                if done_idx % 10 == 0 or rec["empty_subtree_actions"] or rec["header_noise_actions"] or not rec["success"] or rec["action_count"] == 0:
                    print(
                        f"[{done_idx}/{len(bug_keys)}] {bk}: ok={rec['success']}, actions={rec['action_count']}, "
                        f"empty_sub={len(rec['empty_subtree_actions'])}, header={len(rec['header_noise_actions'])}",
                        flush=True,
                    )

    results = [results_map[bk] for bk in bug_keys]
    zero_actions = [r["bug_key"] for r in results if not r["success"] or r["action_count"] == 0]
    empty_subtrees = [{"bug_key": r["bug_key"], "actions": r["empty_subtree_actions"]} for r in results if r["empty_subtree_actions"]]
    header_noise_hits = [{"bug_key": r["bug_key"], "actions": r["header_noise_actions"]} for r in results if r["header_noise_actions"]]

    report = {
        "total_audited": len(results),
        "all_succeeded": len(zero_actions) == 0,
        "bugs_with_zero_actions": zero_actions,
        "bugs_with_empty_subtree_summary": empty_subtrees,
        "bugs_with_header_noise": header_noise_hits,
        "per_bug": results,
    }
    Path("data/normalizer_audit_80.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"DONE: audited={len(results)}, zero_actions={len(zero_actions)}, "
        f"empty_subtree_bugs={len(empty_subtrees)}, header_noise_bugs={len(header_noise_hits)}"
    )


if __name__ == "__main__":
    main()
