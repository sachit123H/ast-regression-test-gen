"""End-to-end post-loop evaluation pipeline (Oracle -> Cobertura Spectra -> FDG/SBFL).

Implements Proposal Sections 4.5 and 4.6:
1. Runs the isolated Differential Execution Oracle once per `(bug, condition)`.
2. For tests satisfying the differential oracle, collects pre-fix Cobertura
   coverage over `classes.relevant` for:
   - Baseline suite `T` (Defects4J `tests.trigger` failing tests)
   - Candidate BRT `t` (injected into a clean pre-fix copy)
3. Persists raw coverage spectra (`data/results/spectra/<bug_key>.json`) so
   DStar (`*=2`) robustness checks are computed with ZERO extra test execution.
4. Computes `Split`, `Cover`, `FDG` (`alpha=0.5`), baseline & augmented method
   ranks, `delta acc@n` (`n in {1,3,5,10}`), and `delta mAP` under BOTH Ochiai
   (primary) and DStar (`*=2`).
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from src.config import ExperimentConfig
from src.evaluation.cobertura_parser import (
    CoberturaCoverage,
    build_coverage_matrix,
    parse_cobertura_xml,
)
from src.evaluation.fdg import compute_fdg
from src.evaluation.oracle import evaluate_differential_oracle
from src.evaluation.sbfl import (
    FormulaName,
    aggregate_line_to_method_scores,
    average_precision,
    best_faulty_rank,
    compute_suspiciousness,
    rank_with_max_tiebreaker,
)
from src.mcp.sandbox import get_d4j_env, wrap_d4j_cmd


def _measure_single_test_cobertura(
    prefix_work_dir: Path,
    fqn_test: str,
    relevant_classes: List[str],
    d4j_bin: str = "defects4j",
    timeout: int = 180,
    is_chart: bool = False,
    require_pass: bool = False,
) -> Optional[CoberturaCoverage]:
    """Run `defects4j coverage -t <fqn_test>` on pre-fix checkout and parse `coverage.xml`."""
    cov_xml = prefix_work_dir / "coverage.xml"
    if cov_xml.exists():
        cov_xml.unlink()
    failing_log = prefix_work_dir / "failing_tests"
    if require_pass and failing_log.exists():
        failing_log.unlink()
    cmd = wrap_d4j_cmd([d4j_bin, "coverage", "-t", fqn_test], is_chart=is_chart)
    proc = subprocess.run(
        cmd,
        cwd=str(prefix_work_dir),
        capture_output=True,
        text=True,
        errors="replace",
        timeout=timeout,
        env=get_d4j_env(),
    )
    if proc.returncode != 0 or not cov_xml.exists():
        return None
    if require_pass:
        combined = proc.stdout + "\n" + proc.stderr
        if "WARNING: Some tests failed" in combined:
            return None
        if failing_log.exists() and failing_log.read_text(encoding="utf-8", errors="replace").strip():
            return None
    return parse_cobertura_xml(cov_xml, relevant_classes=relevant_classes)


def extract_relevant_passing_test_fqns(
    prefix_work_dir: Path,
    dir_src_tests: str,
    triggering_tests: List[str],
    d4j_bin: str = "defects4j",
    max_tests: int = 25,
    is_chart: bool = False,
    target_class_fqn: str = "",
) -> List[str]:
    """Discover non-triggering test methods sampled across `tests.relevant` classes."""
    if max_tests <= 0:
        return []
    proc = subprocess.run(
        wrap_d4j_cmd([d4j_bin, "export", "-p", "tests.relevant", "-w", str(prefix_work_dir)], is_chart=is_chart),
        capture_output=True,
        text=True,
        errors="replace",
        timeout=60,
        env=get_d4j_env(),
    )
    if proc.returncode != 0:
        return []

    trig_set = {t.strip() for t in triggering_tests}
    trig_classes = {t.split("::", 1)[0].strip() for t in triggering_tests if "::" in t}
    target_simple = target_class_fqn.split(".")[-1] if target_class_fqn else ""

    rel_classes = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
    # Prioritize classes matching triggering test class or target class name, preserving stability
    def _class_priority(cls_fqn: str) -> Tuple[int, str]:
        if cls_fqn in trig_classes:
            return (0, cls_fqn)
        if target_simple and target_simple in cls_fqn.split(".")[-1]:
            return (1, cls_fqn)
        return (2, cls_fqn)

    ordered_classes = sorted(rel_classes, key=_class_priority)
    method_re = re.compile(
        r"^\s*(?:@Test(?:\s*\([^)]*\))?\s+)?public\s+void\s+([A-Za-z_]\w*)\s*\(\s*\)",
        re.MULTILINE,
    )

    per_class_methods: List[List[str]] = []
    seen: Set[str] = set()
    for cls_fqn in ordered_classes:
        rel_test_file = prefix_work_dir / f"{dir_src_tests}/{cls_fqn.replace('.', '/')}.java"
        if not rel_test_file.exists():
            continue
        src = rel_test_file.read_text(encoding="utf-8", errors="replace")
        bucket: List[str] = []
        for m in method_re.finditer(src):
            m_name = m.group(1)
            prefix_slice = src[max(0, m.start() - 40) : m.start()]
            if "@Test" not in m.group(0) and "@Test" not in prefix_slice and not m_name.startswith("test"):
                continue
            fqn = f"{cls_fqn}::{m_name}"
            if fqn in trig_set or fqn in seen:
                continue
            seen.add(fqn)
            bucket.append(fqn)
        if bucket:
            per_class_methods.append(bucket)

    # Round-robin across relevant test classes so baseline passing tests cover diverse modules
    passing_candidates: List[str] = []
    round_idx = 0
    while len(passing_candidates) < max_tests:
        progressed = False
        for bucket in per_class_methods:
            if round_idx < len(bucket):
                passing_candidates.append(bucket[round_idx])
                progressed = True
                if len(passing_candidates) >= max_tests:
                    break
        if not progressed:
            break
        round_idx += 1
    return passing_candidates


def _is_test_passing_on_prefix(
    prefix_work_dir: Path,
    fqn_test: str,
    d4j_bin: str = "defects4j",
    timeout: int = 120,
    is_chart: bool = False,
) -> bool:
    """Confirm a candidate baseline test passes cleanly on pre-fix before labeling R(t)=1."""
    proc = subprocess.run(
        wrap_d4j_cmd([d4j_bin, "test", "-t", fqn_test], is_chart=is_chart),
        cwd=str(prefix_work_dir),
        capture_output=True,
        text=True,
        errors="replace",
        timeout=timeout,
        env=get_d4j_env(),
    )
    return proc.returncode == 0 and "Failing tests: 0" in proc.stdout


def _find_faulty_methods(
    target_class_fqn: str,
    enclosing_method_name: str,
    line_to_method: Dict[str, str],
    method_start: int = 1,
    method_end: int = 1,
) -> Set[str]:
    """Match the ground-truth faulty method key(s) in Cobertura's method naming."""
    matched: Set[str] = set()
    simple_class = target_class_fqn.split(".")[-1].split("$")[-1]
    candidate_names = {enclosing_method_name}
    if enclosing_method_name == simple_class or enclosing_method_name == "<init>":
        candidate_names.add("<init>")
    if enclosing_method_name in ("<class-level>", "<clinit>", "<static>"):
        candidate_names.add("<clinit>")

    for m_key in set(line_to_method.values()):
        if "#" not in m_key:
            continue
        c_name, m_sig = m_key.split("#", 1)
        if c_name == target_class_fqn or c_name.startswith(f"{target_class_fqn}$"):
            for cand in candidate_names:
                if m_sig.startswith(f"{cand}(") or m_sig == cand:
                    matched.add(m_key)

    # Fallback: if name matching didn't catch it, match by line range in target class
    if not matched and method_start <= method_end:
        for ln in range(method_start, method_end + 1):
            elem = f"{target_class_fqn}:{ln}"
            if elem in line_to_method:
                matched.add(line_to_method[elem])

    return matched


def evaluate_spectra_for_formula(
    elements: List[str],
    baseline_matrix: np.ndarray,
    baseline_results: List[int],
    candidate_vector: List[int],
    line_to_method: Dict[str, str],
    faulty_methods: Set[str],
    formula: FormulaName,
    alpha: float = 0.5,
    dstar_star: int = 2,
) -> Dict[str, Any]:
    """Compute FDG + baseline/augmented SBFL ranks & AP from stored spectra (zero test re-runs)."""
    fdg_res = compute_fdg(
        baseline_matrix=baseline_matrix,
        baseline_results=baseline_results,
        new_test_coverage=candidate_vector,
        alpha=alpha,
        formula=formula,
        dstar_star=dstar_star,
    )

    # Baseline SBFL ranking (T)
    base_scores = compute_suspiciousness(
        baseline_matrix, baseline_results, formula=formula, dstar_star=dstar_star
    )
    base_method_scores = aggregate_line_to_method_scores(elements, base_scores, line_to_method)
    base_ranks = rank_with_max_tiebreaker(base_method_scores)
    base_best = best_faulty_rank(base_ranks, faulty_methods)
    base_ap = average_precision(base_ranks, faulty_methods)

    # Augmented SBFL ranking (T U {t}, where valid BRT `t` fails on pre-fix -> result 0)
    aug_matrix = np.vstack([baseline_matrix, np.asarray(candidate_vector, dtype=np.int64).reshape(1, -1)])
    aug_results = list(baseline_results) + [0]
    aug_scores = compute_suspiciousness(
        aug_matrix, aug_results, formula=formula, dstar_star=dstar_star
    )
    aug_method_scores = aggregate_line_to_method_scores(elements, aug_scores, line_to_method)
    aug_ranks = rank_with_max_tiebreaker(aug_method_scores)
    aug_best = best_faulty_rank(aug_ranks, faulty_methods)
    aug_ap = average_precision(aug_ranks, faulty_methods)

    return {
        "formula": formula,
        "split": fdg_res.split,
        "cover": fdg_res.cover,
        "fdg": fdg_res.fdg,
        "ag_before": fdg_res.num_ambiguity_groups_before,
        "ag_after": fdg_res.num_ambiguity_groups_after,
        "baseline_best_rank": base_best,
        "augmented_best_rank": aug_best,
        "rank_improvement": base_best - aug_best,
        "baseline_ap": round(base_ap, 6),
        "augmented_ap": round(aug_ap, 6),
        "delta_ap": round(aug_ap - base_ap, 6),
    }


def _ensure_d4j_checkout(
    project: str,
    bug_id: int,
    version_tag: str,
    target_dir: Path,
    d4j_bin: str = "defects4j",
    is_chart: bool = False,
) -> bool:
    """Checkout `<bug_id><version_tag>` into `target_dir` if not already present."""
    if target_dir.exists():
        return False
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        wrap_d4j_cmd(
            [d4j_bin, "checkout", "-p", project, "-v", f"{bug_id}{version_tag}", "-w", str(target_dir)],
            is_chart=is_chart,
        ),
        capture_output=True,
        text=True,
        errors="replace",
        timeout=180,
        env=get_d4j_env(),
    )
    if proc.returncode != 0:
        shutil.rmtree(target_dir, ignore_errors=True)
        raise RuntimeError(f"defects4j checkout {project}-{bug_id}{version_tag} failed: {proc.stderr.strip()}")
    return True


def _evaluate_single_bug(
    bug: Dict[str, Any],
    cfg: ExperimentConfig,
    trials_dir: Path,
    spectra_dir: Path,
    auto_checkout: bool = True,
) -> Dict[str, Any]:
    """Evaluate differential oracle + full-suite Cobertura FDG spectra for a single bug."""
    bug_key = bug["bug_key"]
    project = bug["project"]
    bug_id = int(bug.get("bug_id") or bug_key.split("-")[1])
    is_chart = project == "Chart" or "org.jfree" in bug.get("target_rel_file", "")

    raw_pre_dir = cfg.paths.prefix_root / f"{bug_key}_raw"
    post_dir = cfg.paths.postfix_root / bug_key
    created_pre = False
    created_post = False

    try:
        bug_trials: Dict[str, Dict[str, Any]] = {}
        for cond in ("raw_diff", "ast_edit"):
            trial_path = trials_dir / f"{bug_key}__{cond}.json"
            if trial_path.exists():
                bug_trials[cond] = json.loads(trial_path.read_text(encoding="utf-8"))

        needs_postfix = any(
            t.get("final_java_source") and t.get("terminated_reason") == "PREFIX_FAIL_REACHED"
            for t in bug_trials.values()
        )
        if auto_checkout and needs_postfix and not post_dir.exists():
            created_post = _ensure_d4j_checkout(
                project, bug_id, "f", post_dir, d4j_bin=cfg.tools.defects4j_bin, is_chart=is_chart
            )

        verdicts: Dict[str, Dict[str, Any]] = {}
        for cond in ("raw_diff", "ast_edit"):
            if cond not in bug_trials:
                continue
            verdict = evaluate_differential_oracle(
                bug=bug,
                trial=bug_trials[cond],
                postfix_checkout_dir=post_dir,
                d4j_bin=cfg.tools.defects4j_bin,
                compile_timeout=cfg.tools.compile_timeout_sec,
                test_timeout=cfg.tools.test_timeout_sec,
            )
            verdicts[cond] = verdict.to_dict()

        valid_conds = [
            c
            for c, v in verdicts.items()
            if (
                v["is_valid_strict_assertion"]
                if cfg.evaluation.strict_assertion_only
                else v["is_valid_libro"]
            )
        ]

        fdg_by_cond: Dict[str, Dict[str, Any]] = {}
        if valid_conds:
            if auto_checkout and not raw_pre_dir.exists():
                created_pre = _ensure_d4j_checkout(
                    project, bug_id, "b", raw_pre_dir, d4j_bin=cfg.tools.defects4j_bin, is_chart=is_chart
                )

            if raw_pre_dir.exists():
                base_covs: List[CoberturaCoverage] = []
                base_results: List[int] = []
                for trig in bug["triggering_tests"]:
                    cov = _measure_single_test_cobertura(
                        raw_pre_dir,
                        trig,
                        bug["relevant_classes"],
                        d4j_bin=cfg.tools.defects4j_bin,
                        timeout=cfg.tools.coverage_timeout_sec,
                        is_chart=is_chart,
                    )
                    if cov is not None:
                        base_covs.append(cov)
                        base_results.append(0)

                if cfg.evaluation.include_passing_baseline_tests:
                    pass_fqns = extract_relevant_passing_test_fqns(
                        raw_pre_dir,
                        bug["dir_src_tests"],
                        bug["triggering_tests"],
                        d4j_bin=cfg.tools.defects4j_bin,
                        max_tests=cfg.evaluation.max_baseline_passing_tests,
                        is_chart=is_chart,
                        target_class_fqn=bug["modified_classes"][0],
                    )
                    for pfqn in pass_fqns:
                        pcov = _measure_single_test_cobertura(
                            raw_pre_dir,
                            pfqn,
                            bug["relevant_classes"],
                            d4j_bin=cfg.tools.defects4j_bin,
                            timeout=cfg.tools.coverage_timeout_sec,
                            is_chart=is_chart,
                            require_pass=True,
                        )
                        if pcov is not None:
                            base_covs.append(pcov)
                            base_results.append(1)

                if base_covs:
                    elements, base_mat_list, line_to_method = build_coverage_matrix(base_covs)
                    base_mat = np.asarray(base_mat_list, dtype=np.int64)
                    faulty_methods = _find_faulty_methods(
                        bug["modified_classes"][0],
                        bug["enclosing_method"],
                        line_to_method,
                        method_start=int(bug.get("enclosing_method_start", 1)),
                        method_end=int(bug.get("enclosing_method_end", 1)),
                    )

                    stored_spectra: Dict[str, Any] = {
                        "bug_key": bug_key,
                        "elements": elements,
                        "line_to_method": line_to_method,
                        "faulty_methods": sorted(faulty_methods),
                        "baseline_matrix": base_mat_list,
                        "baseline_results": base_results,
                        "num_baseline_failing": sum(1 for r in base_results if r == 0),
                        "num_baseline_passing": sum(1 for r in base_results if r == 1),
                        "candidates": {},
                    }

                    for cond in valid_conds:
                        trial = bug_trials[cond]
                        cov_work = cfg.paths.prefix_root / f"{bug_key}_cov_{cond}"
                        if cov_work.exists():
                            shutil.rmtree(cov_work)
                        shutil.copytree(raw_pre_dir, cov_work, symlinks=False)
                        try:
                            pkg = trial["final_package_name"]
                            cls_name = trial["final_test_class_name"]
                            m_name = trial["final_test_method_name"]
                            pkg_path = pkg.replace(".", "/") if pkg else ""
                            rel_t = (
                                f"{bug['dir_src_tests']}/{pkg_path}/{cls_name}.java"
                                if pkg_path
                                else f"{bug['dir_src_tests']}/{cls_name}.java"
                            )
                            t_file = cov_work / rel_t
                            t_file.parent.mkdir(parents=True, exist_ok=True)
                            t_file.write_text(trial["final_java_source"], encoding="utf-8")

                            d4j_env = get_d4j_env()
                            comp_cov = subprocess.run(
                                wrap_d4j_cmd([cfg.tools.defects4j_bin, "compile"], is_chart=is_chart),
                                cwd=str(cov_work),
                                capture_output=True,
                                text=True,
                                errors="replace",
                                timeout=cfg.tools.compile_timeout_sec,
                                env=d4j_env,
                            )
                            if comp_cov.returncode != 0:
                                continue

                            fqn = f"{pkg}.{cls_name}::{m_name}" if pkg else f"{cls_name}::{m_name}"
                            cand_cov = _measure_single_test_cobertura(
                                cov_work,
                                fqn,
                                bug["relevant_classes"],
                                d4j_bin=cfg.tools.defects4j_bin,
                                timeout=cfg.tools.coverage_timeout_sec,
                                is_chart=is_chart,
                            )
                            if cand_cov is not None:
                                cand_vec = [1 if e in cand_cov.covered_lines else 0 for e in elements]
                                stored_spectra["candidates"][cond] = cand_vec
                                ochiai_eval = evaluate_spectra_for_formula(
                                    elements=elements,
                                    baseline_matrix=base_mat,
                                    baseline_results=base_results,
                                    candidate_vector=cand_vec,
                                    line_to_method=line_to_method,
                                    faulty_methods=faulty_methods,
                                    formula="ochiai",
                                    alpha=cfg.evaluation.fdg_alpha,
                                )
                                dstar_eval = evaluate_spectra_for_formula(
                                    elements=elements,
                                    baseline_matrix=base_mat,
                                    baseline_results=base_results,
                                    candidate_vector=cand_vec,
                                    line_to_method=line_to_method,
                                    faulty_methods=faulty_methods,
                                    formula="dstar",
                                    alpha=cfg.evaluation.fdg_alpha,
                                    dstar_star=cfg.evaluation.dstar_star,
                                )
                                fdg_by_cond[cond] = {"ochiai": ochiai_eval, "dstar": dstar_eval}
                        finally:
                            if cov_work.exists():
                                shutil.rmtree(cov_work, ignore_errors=True)

                    (spectra_dir / f"{bug_key}.json").write_text(
                        json.dumps(stored_spectra, indent=2), encoding="utf-8"
                    )

        return {
            "bug_key": bug_key,
            "project": project,
            "verdicts": verdicts,
            "fdg": fdg_by_cond,
        }
    finally:
        if created_pre and raw_pre_dir.exists():
            shutil.rmtree(raw_pre_dir, ignore_errors=True)
        if created_post and post_dir.exists():
            shutil.rmtree(post_dir, ignore_errors=True)


def run_evaluation_pipeline(
    cfg: ExperimentConfig,
    workers: int = 1,
    auto_checkout: bool = True,
    scratch_root: Optional[Path] = None,
    bug_keys: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Run differential oracle + Cobertura spectra + FDG/SBFL over all completed trials."""
    if scratch_root is not None:
        cfg.paths.prefix_root = scratch_root / "prefix"
        cfg.paths.postfix_root = scratch_root / "postfix"
    all_bugs = json.loads(cfg.paths.curated_manifest.read_text(encoding="utf-8"))
    bugs = [b for b in all_bugs if b["bug_key"] in set(bug_keys)] if bug_keys else all_bugs
    trials_dir = cfg.paths.results_dir / "trials"
    spectra_dir = cfg.paths.results_dir / "spectra"
    spectra_dir.mkdir(parents=True, exist_ok=True)

    out_json = cfg.paths.results_dir / "evaluation_summary.json"
    records_by_key: Dict[str, Dict[str, Any]] = {}
    if bug_keys and out_json.exists():
        for existing in json.loads(out_json.read_text(encoding="utf-8")):
            records_by_key[existing["bug_key"]] = existing

    if workers <= 1:
        for idx, bug in enumerate(bugs, start=1):
            bk = bug["bug_key"]
            print(f"[{idx}/{len(bugs)}] Evaluating {bk} ({bug['project']})...", flush=True)
            rec = _evaluate_single_bug(bug, cfg, trials_dir, spectra_dir, auto_checkout=auto_checkout)
            records_by_key[bk] = rec
            for c, v in rec["verdicts"].items():
                print(f"  -> {c}: {v['classification']} ({v['outcome_pair']})", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {
                pool.submit(_evaluate_single_bug, bug, cfg, trials_dir, spectra_dir, auto_checkout): bug["bug_key"]
                for bug in bugs
            }
            for done_idx, fut in enumerate(as_completed(futs), start=1):
                rec = fut.result()
                bk = rec["bug_key"]
                records_by_key[bk] = rec
                summary_str = ", ".join(f"{c}={v['classification']}" for c, v in rec["verdicts"].items())
                print(f"[{done_idx}/{len(bugs)}] {bk}: {summary_str}", flush=True)

    all_records = [records_by_key[b["bug_key"]] for b in all_bugs if b["bug_key"] in records_by_key]

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(all_records, indent=2), encoding="utf-8")

    # Persist evaluation metadata alongside summary so configs and outputs never diverge
    eval_meta = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "total_bugs_evaluated": len(all_records),
        "evaluation_config": {
            "fdg_alpha": cfg.evaluation.fdg_alpha,
            "dstar_star": cfg.evaluation.dstar_star,
            "strict_assertion_only": cfg.evaluation.strict_assertion_only,
            "include_passing_baseline_tests": cfg.evaluation.include_passing_baseline_tests,
            "max_baseline_passing_tests": cfg.evaluation.max_baseline_passing_tests,
        },
    }
    (cfg.paths.results_dir / "evaluation_metadata.json").write_text(
        json.dumps(eval_meta, indent=2), encoding="utf-8"
    )

    # Export flat CSV for easy inspection & statistical plotting
    out_csv = cfg.paths.results_dir / "evaluation_paired.csv"
    fieldnames = [
        "bug_key",
        "project",
        "raw_valid_libro",
        "ast_valid_libro",
        "raw_valid_strict",
        "ast_valid_strict",
        "raw_classification",
        "ast_classification",
        "raw_fdg_ochiai",
        "ast_fdg_ochiai",
        "raw_fdg_dstar",
        "ast_fdg_dstar",
        "raw_delta_ap_ochiai",
        "ast_delta_ap_ochiai",
        "raw_rank_imp_ochiai",
        "ast_rank_imp_ochiai",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for rec in all_records:
            vr = rec["verdicts"].get("raw_diff", {})
            va = rec["verdicts"].get("ast_edit", {})
            fr_o = rec["fdg"].get("raw_diff", {}).get("ochiai", {})
            fa_o = rec["fdg"].get("ast_edit", {}).get("ochiai", {})
            fr_d = rec["fdg"].get("raw_diff", {}).get("dstar", {})
            fa_d = rec["fdg"].get("ast_edit", {}).get("dstar", {})
            writer.writerow(
                {
                    "bug_key": rec["bug_key"],
                    "project": rec["project"],
                    "raw_valid_libro": vr.get("is_valid_libro", False),
                    "ast_valid_libro": va.get("is_valid_libro", False),
                    "raw_valid_strict": vr.get("is_valid_strict_assertion", False),
                    "ast_valid_strict": va.get("is_valid_strict_assertion", False),
                    "raw_classification": vr.get("classification", "MISSING"),
                    "ast_classification": va.get("classification", "MISSING"),
                    "raw_fdg_ochiai": fr_o.get("fdg", ""),
                    "ast_fdg_ochiai": fa_o.get("fdg", ""),
                    "raw_fdg_dstar": fr_d.get("fdg", ""),
                    "ast_fdg_dstar": fa_d.get("fdg", ""),
                    "raw_delta_ap_ochiai": fr_o.get("delta_ap", ""),
                    "ast_delta_ap_ochiai": fa_o.get("delta_ap", ""),
                    "raw_rank_imp_ochiai": fr_o.get("rank_improvement", ""),
                    "ast_rank_imp_ochiai": fa_o.get("rank_improvement", ""),
                }
            )

    return {"total_bugs_evaluated": len(all_records), "summary_path": str(out_json), "csv_path": str(out_csv)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run differential oracle and FDG/SBFL evaluation.")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--workers", type=int, default=1, help="Parallel worker threads across bugs")
    parser.add_argument("--no-auto-checkout", action="store_true", help="Skip auto-checkout of missing D4J dirs")
    parser.add_argument("--scratch-root", type=Path, default=None, help="Fast local scratch dir for D4J checkouts (e.g. /tmp/d4j_eval)")
    parser.add_argument("--bug-keys", nargs="+", default=None, help="Optional subset of bug keys to re-evaluate and merge")
    args = parser.parse_args()
    cfg = ExperimentConfig.from_yaml(args.config)
    res = run_evaluation_pipeline(
        cfg,
        workers=args.workers,
        auto_checkout=not args.no_auto_checkout,
        scratch_root=args.scratch_root,
        bug_keys=args.bug_keys,
    )
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
