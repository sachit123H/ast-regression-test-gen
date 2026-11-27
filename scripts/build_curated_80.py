"""Build the full 80-bug rerun manifest (`data/curated_bugs_80.json`) using the patched GumTree normalizer and multi-line enclosing method extraction."""
from __future__ import annotations

import json
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ast_diff.gumtree_normalizer import extract_raw_git_diff, run_gumtree_diff
from src.config import ExperimentConfig
from src.dataset.defects4j_curate import (
    CuratedBug,
    MethodSpan,
    _export_prop,
    _run_d4j,
    find_enclosing_methods,
    parse_diff_changed_lines,
)


def _curate_one(bk: str, cfg: ExperimentConfig, base_tmp: Path) -> CuratedBug:
    proj, bid_str = bk.split("-")
    bid = int(bid_str)
    pre_dir = base_tmp / f"{bk}_b"
    post_dir = base_tmp / f"{bk}_f"
    try:
        _run_d4j(["checkout", "-p", proj, "-v", f"{bid}b", "-w", str(pre_dir)], cfg.tools.defects4j_bin, project=proj)
        _run_d4j(["checkout", "-p", proj, "-v", f"{bid}f", "-w", str(post_dir)], cfg.tools.defects4j_bin, project=proj)

        triggering = _export_prop(cfg.tools.defects4j_bin, pre_dir, "tests.trigger", project=proj)
        modified_classes = _export_prop(cfg.tools.defects4j_bin, pre_dir, "classes.modified", project=proj)
        relevant_classes = _export_prop(cfg.tools.defects4j_bin, pre_dir, "classes.relevant", project=proj)
        src_dirs = _export_prop(cfg.tools.defects4j_bin, pre_dir, "dir.src.classes", project=proj)
        test_dirs = _export_prop(cfg.tools.defects4j_bin, pre_dir, "dir.src.tests", project=proj)
        dir_src_classes = src_dirs[0] if src_dirs else "src/main/java"
        dir_src_tests = test_dirs[0] if test_dirs else "src/test/java"
        rel_java = f"{dir_src_classes}/{modified_classes[0].replace('.', '/')}.java"

        pre_java = pre_dir / rel_java
        post_java = post_dir / rel_java
        raw_diff = extract_raw_git_diff(pre_java, post_java, rel_java)
        pre_source = pre_java.read_text(encoding="utf-8", errors="replace")
        changed_lines = parse_diff_changed_lines(raw_diff)
        methods = find_enclosing_methods(pre_source, changed_lines)
        primary = methods[0] if methods else MethodSpan(name="<class-level>", signature="", start_line=1, end_line=1)

        ast_res = run_gumtree_diff(
            src_file=pre_java,
            dst_file=post_java,
            rel_path=rel_java,
            enclosing_method=primary.name,
            gumtree_bin=cfg.tools.gumtree_bin,
            java_home=cfg.tools.gumtree_java_home,
            filter_header_noise=cfg.dataset.filter_ast_header_noise,
        )

        pre_lines = pre_source.splitlines()
        m_start = max(1, primary.start_line)
        m_end = min(len(pre_lines), max(m_start, primary.end_line), m_start + 249)
        enclosing_src = (
            "\n".join(f"{ln:4d} | {pre_lines[ln - 1]}" for ln in range(m_start, m_end + 1))
            if pre_lines and primary.end_line >= primary.start_line
            else ""
        )

        return CuratedBug(
            bug_key=bk,
            project=proj,
            bug_id=bid,
            modified_classes=modified_classes,
            relevant_classes=relevant_classes,
            triggering_tests=triggering,
            dir_src_classes=dir_src_classes,
            dir_src_tests=dir_src_tests,
            target_rel_file=rel_java,
            enclosing_method=primary.name,
            enclosing_method_start=primary.start_line,
            enclosing_method_end=primary.end_line,
            raw_diff=raw_diff,
            ast_edit_script=ast_res.normalized_text,
            ast_action_count=ast_res.raw_action_count,
            is_single_method_fix=len(methods) == 1,
            enclosing_method_source=enclosing_src,
        )
    finally:
        shutil.rmtree(pre_dir, ignore_errors=True)
        shutil.rmtree(post_dir, ignore_errors=True)


def main() -> None:
    cfg = ExperimentConfig.from_yaml("configs/experiment.yaml")
    pilot_65 = json.loads(Path("data/curated_bugs.json").read_text(encoding="utf-8"))
    excl = json.loads(Path("data/exclusion_log.json").read_text(encoding="utf-8"))
    bug_keys = [b["bug_key"] for b in pilot_65] + [
        item["bug_key"] for item in excl.get("api_quota_exhausted_aborted_bugs", [])
    ]

    records_map = {}
    with tempfile.TemporaryDirectory(prefix="curate80_") as tmp:
        tmp_dir = Path(tmp)
        with ThreadPoolExecutor(max_workers=6) as pool:
            futs = {pool.submit(_curate_one, bk, cfg, tmp_dir): bk for bk in bug_keys}
            for idx, fut in enumerate(as_completed(futs), start=1):
                rec = fut.result()
                records_map[rec.bug_key] = asdict(rec)
                if idx % 10 == 0:
                    print(f"[{idx}/{len(bug_keys)}] Curated {rec.bug_key} (inline_len={len(rec.enclosing_method_source)})", flush=True)

    ordered = [records_map[bk] for bk in bug_keys]
    cfg.paths.curated_manifest.write_text(json.dumps(ordered, indent=2), encoding="utf-8")

    # Write clean 80-bug pre-rerun exclusion log (no API quota drops)
    excl_80 = {
        "total_scanned": excl["total_scanned"],
        "total_curated": len(ordered),
        "localized_filter_relaxed": False,
        "exclusion_counts": {
            "checkout_or_build_failed": excl["exclusion_counts"]["checkout_or_build_failed"],
            "multi_file_or_multi_method": excl["exclusion_counts"]["multi_file_or_multi_method"],
            "gumtree_parse_failed": excl["exclusion_counts"]["gumtree_parse_failed"],
            "no_triggering_tests": excl["exclusion_counts"]["no_triggering_tests"],
        },
        "project_breakdown": {
            "Commons-Lang": {"scanned": 65, "curated": 40, "multi_method": 22, "gumtree_failed": 3},
            "Commons-Math": {"scanned": 74, "curated": 40, "multi_method": 34, "gumtree_failed": 0},
        },
        "gumtree_parse_failed_bugs": excl.get("gumtree_parse_failed_bugs", []),
    }
    cfg.paths.exclusion_log.write_text(json.dumps(excl_80, indent=2), encoding="utf-8")
    print(f"Saved {len(ordered)} curated bugs -> {cfg.paths.curated_manifest}")


if __name__ == "__main__":
    main()
