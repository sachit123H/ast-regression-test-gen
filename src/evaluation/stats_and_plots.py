"""Statistical hypothesis testing, effect sizes, LaTeX tables, and publication figures.

Implements COINSE / empirical SE statistical rigor:
- RQ1 (Paired Binary Validity): Exact McNemar's test (binomial on discordant pairs).
- RQ2 (Paired FDG & Localisation Gain on jointly valid subset):
  - Wilcoxon signed-rank test (two-sided)
  - Vargha-Delaney A12 effect size (standard in SBSE / Shin Yoo's papers)
  - Separate reporting for Ochiai (primary) and DStar (*=2, robustness)
  - acc@1, acc@3, acc@5, acc@10 and mAP before vs. after test addition
- Outputs ready-to-include LaTeX tables and 300-DPI matplotlib figures.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from src.config import ExperimentConfig


def mcnemar_exact(raw_valid: Sequence[bool], ast_valid: Sequence[bool]) -> Dict[str, Any]:
    """Exact two-sided McNemar test on paired binary outcomes."""
    both_valid = sum(1 for r, a in zip(raw_valid, ast_valid) if r and a)
    raw_only = sum(1 for r, a in zip(raw_valid, ast_valid) if r and not a)
    ast_only = sum(1 for r, a in zip(raw_valid, ast_valid) if not r and a)
    both_invalid = sum(1 for r, a in zip(raw_valid, ast_valid) if not r and not a)

    discordant = raw_only + ast_only
    if discordant == 0:
        p_val = 1.0
    else:
        p_val = float(stats.binomtest(min(raw_only, ast_only), discordant, 0.5).pvalue)

    return {
        "n_total": len(raw_valid),
        "both_valid": both_valid,
        "raw_only_valid": raw_only,
        "ast_only_valid": ast_only,
        "both_invalid": both_invalid,
        "raw_rate": round((both_valid + raw_only) / max(1, len(raw_valid)), 4),
        "ast_rate": round((both_valid + ast_only) / max(1, len(raw_valid)), 4),
        "p_value": round(p_val, 6),
    }


def vargha_delaney_a12(treatment: Sequence[float], control: Sequence[float]) -> float:
    """Compute Vargha-Delaney A12 effect size P(treatment > control) + 0.5*P(tie).

    Interpretation (Vargha & Delaney, 2000):
      0.50 = no effect; >=0.56 small; >=0.64 medium; >=0.71 large.
    """
    x = list(treatment)
    y = list(control)
    if not x or not y:
        return 0.5
    greater = 0.0
    ties = 0.0
    for xi in x:
        for yi in y:
            if xi > yi:
                greater += 1.0
            elif xi == yi:
                ties += 1.0
    return (greater + 0.5 * ties) / (len(x) * len(y))


def compute_tost_equivalence(
    raw_vals: Sequence[float],
    ast_vals: Sequence[float],
    margin: float = 0.01,
) -> Dict[str, Any]:
    """Two One-Sided Tests (TOST) for paired difference equivalence within [-margin, +margin]."""
    r = np.asarray(raw_vals, dtype=np.float64)
    a = np.asarray(ast_vals, dtype=np.float64)
    diffs = a - r
    n = len(diffs)
    if n < 2:
        return {"tost_p_value": 1.0, "equivalent": False}

    mean_diff = float(np.mean(diffs))
    sd_diff = float(np.std(diffs, ddof=1))
    se = float(sd_diff / np.sqrt(n))
    df = n - 1

    if se == 0.0:
        in_band = abs(mean_diff) < margin
        t1, t2 = (float("inf"), float("inf")) if in_band else (0.0, 0.0)
        p1, p2 = (0.0, 0.0) if in_band else (1.0, 1.0)
    else:
        t1 = (mean_diff - (-margin)) / se
        t2 = (margin - mean_diff) / se
        p1 = float(stats.t.sf(t1, df=df))
        p2 = float(stats.t.sf(t2, df=df))
    p_tost = max(p1, p2)

    n_exact_ties = int(np.sum(np.isclose(diffs, 0.0, atol=1e-9)))
    n_positive = int(np.sum(diffs > 1e-9))
    n_negative = int(np.sum(diffs < -1e-9))
    tie_fraction = round(n_exact_ties / n, 4)

    return {
        "n_paired": n,
        "equivalence_margin": margin,
        "mean_diff": round(mean_diff, 6),
        "sd_diff": round(sd_diff, 6),
        "se_diff": round(se, 6),
        "t1_stat": round(t1, 4),
        "t2_stat": round(t2, 4),
        "p1": p1,
        "p2": p2,
        "p_tost": p_tost,
        "equivalent_at_005": bool(p_tost < 0.05),
        "n_exact_ties": n_exact_ties,
        "n_positive_diffs": n_positive,
        "n_negative_diffs": n_negative,
        "zero_tie_fraction": tie_fraction,
        "parametric_caution": bool(tie_fraction >= 0.5),
    }


def compute_exact_sign_test(
    raw_vals: Sequence[float],
    ast_vals: Sequence[float],
) -> Dict[str, Any]:
    """Exact two-sided binomial sign test over non-zero paired differences."""
    r = np.asarray(raw_vals, dtype=np.float64)
    a = np.asarray(ast_vals, dtype=np.float64)
    diffs = a - r
    n_pos = int(np.sum(diffs > 1e-9))
    n_neg = int(np.sum(diffs < -1e-9))
    n_ties = int(np.sum(np.isclose(diffs, 0.0, atol=1e-9)))
    discordant = n_pos + n_neg
    p_val = float(stats.binomtest(min(n_pos, n_neg), discordant, 0.5).pvalue) if discordant > 0 else 1.0
    return {
        "n_total": int(len(diffs)),
        "n_positive_ast": n_pos,
        "n_negative_raw": n_neg,
        "n_exact_ties": n_ties,
        "p_value_two_sided": round(p_val, 6),
    }


def compute_bca_bootstrap_ci(
    raw_vals: Sequence[float],
    ast_vals: Sequence[float],
    n_resamples: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Dict[str, Any]:
    """Compute 95% BCa bootstrap confidence interval for mean paired difference."""
    r = np.asarray(raw_vals, dtype=np.float64)
    a = np.asarray(ast_vals, dtype=np.float64)
    diffs = a - r
    if len(diffs) < 2:
        return {"ci_low": 0.0, "ci_high": 0.0}

    res = stats.bootstrap(
        (diffs,),
        np.mean,
        n_resamples=n_resamples,
        confidence_level=1.0 - alpha,
        method="BCa",
        random_state=seed,
    )
    return {
        "confidence_level": 1.0 - alpha,
        "ci_low": round(float(res.confidence_interval.low), 6),
        "ci_high": round(float(res.confidence_interval.high), 6),
        "n_resamples": n_resamples,
    }


def compute_cohen_power(
    n: int = 13,
    d: float = 0.8,
    alpha: float = 0.05,
) -> float:
    """Compute statistical power for paired two-sided t-test at effect size d."""
    df = n - 1
    t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df))
    ncp = float(d * np.sqrt(n))
    power = float(1.0 - stats.nct.cdf(t_crit, df, ncp) + stats.nct.cdf(-t_crit, df, ncp))
    return round(power, 4)


def paired_wilcoxon_summary(
    raw_vals: Sequence[float],
    ast_vals: Sequence[float],
) -> Dict[str, Any]:
    """Run paired Wilcoxon signed-rank test + Vargha-Delaney A12 + descriptive stats."""
    r = np.asarray(raw_vals, dtype=np.float64)
    a = np.asarray(ast_vals, dtype=np.float64)
    n = int(len(r))
    if n == 0:
        return {"n_paired": 0, "p_value": 1.0, "a12": 0.5}

    diffs = a - r
    n_exact_ties = int(np.sum(np.isclose(diffs, 0.0, atol=1e-9)))
    n_pos = int(np.sum(diffs > 1e-9))
    n_neg = int(np.sum(diffs < -1e-9))

    if np.allclose(diffs, 0.0):
        stat, p_val = 0.0, 1.0
    else:
        res = stats.wilcoxon(a, r, zero_method="wilcox", alternative="two-sided")
        stat, p_val = float(res.statistic), float(res.pvalue)

    return {
        "n_paired": n,
        "raw_mean": round(float(np.mean(r)), 4),
        "raw_median": round(float(np.median(r)), 4),
        "ast_mean": round(float(np.mean(a)), 4),
        "ast_median": round(float(np.median(a)), 4),
        "mean_diff_ast_minus_raw": round(float(np.mean(diffs)), 4),
        "wilcoxon_stat": round(stat, 4),
        "p_value": round(p_val, 6),
        "a12": round(vargha_delaney_a12(a.tolist(), r.tolist()), 4),
        "n_exact_ties": n_exact_ties,
        "n_positive_diffs": n_pos,
        "n_negative_diffs": n_neg,
    }


def analyze_evaluation_records(
    records: List[Dict[str, Any]],
    figures_dir: Path,
    trials_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Compute all RQ1, RQ2, and RQ3 summaries, LaTeX tables, and publication plots.

    Strict assertion validity (`VALID_ASSERTION`) is reported as the primary validity
    criterion (Proposal Sec. 4.5), with LIBRO fail/pass (`VALID_ASSERTION` + `VALID_EXCEPTION`)
    reported secondarily. Conditional RQ2 FDG comparisons report simple paired counts
    (wins / ties / losses), exact sign test, paired Wilcoxon, and Vargha-Delaney A12.
    """
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Filter to records where at least one condition was executed (avoids false zeros on pilot/resumed runs)
    executed_records = [
        r for r in records if "raw_diff" in r.get("verdicts", {}) or "ast_edit" in r.get("verdicts", {})
    ]
    eval_records = executed_records if executed_records else records

    raw_strict = [bool(r["verdicts"].get("raw_diff", {}).get("is_valid_strict_assertion", False)) for r in eval_records]
    ast_strict = [bool(r["verdicts"].get("ast_edit", {}).get("is_valid_strict_assertion", False)) for r in eval_records]
    raw_libro = [bool(r["verdicts"].get("raw_diff", {}).get("is_valid_libro", False)) for r in eval_records]
    ast_libro = [bool(r["verdicts"].get("ast_edit", {}).get("is_valid_libro", False)) for r in eval_records]

    rq1_strict = mcnemar_exact(raw_strict, ast_strict)
    rq1_libro = mcnemar_exact(raw_libro, ast_libro)

    def _collect_paired_fdg(strict_only: bool) -> Dict[str, Any]:
        p_ochiai_raw, p_ochiai_ast = [], []
        p_dstar_raw, p_dstar_ast = [], []
        p_dap_raw, p_dap_ast = [], []
        b_ranks, r_ranks, a_ranks = [], [], []
        paired_bugs: List[str] = []

        for r in eval_records:
            vr = r["verdicts"].get("raw_diff", {})
            va = r["verdicts"].get("ast_edit", {})
            if strict_only:
                ok_pair = bool(vr.get("is_valid_strict_assertion", False)) and bool(va.get("is_valid_strict_assertion", False))
            else:
                ok_pair = bool(vr.get("is_valid_libro", False)) and bool(va.get("is_valid_libro", False))
            if not ok_pair:
                continue

            fdg = r.get("fdg", {})
            if "raw_diff" in fdg and "ast_edit" in fdg:
                rd = fdg["raw_diff"]
                ad = fdg["ast_edit"]
                if "ochiai" in rd and "ochiai" in ad:
                    paired_bugs.append(str(r.get("bug_key", "")))
                    p_ochiai_raw.append(float(rd["ochiai"]["fdg"]))
                    p_ochiai_ast.append(float(ad["ochiai"]["fdg"]))
                    p_dap_raw.append(float(rd["ochiai"]["delta_ap"]))
                    p_dap_ast.append(float(ad["ochiai"]["delta_ap"]))
                    if "augmented_best_rank" in rd["ochiai"]:
                        b_ranks.append(int(rd["ochiai"].get("baseline_best_rank", 999)))
                        r_ranks.append(int(rd["ochiai"]["augmented_best_rank"]))
                        a_ranks.append(int(ad["ochiai"]["augmented_best_rank"]))
                if "dstar" in rd and "dstar" in ad:
                    p_dstar_raw.append(float(rd["dstar"]["fdg"]))
                    p_dstar_ast.append(float(ad["dstar"]["fdg"]))

        ochiai_sum = paired_wilcoxon_summary(p_ochiai_raw, p_ochiai_ast)
        dstar_sum = paired_wilcoxon_summary(p_dstar_raw, p_dstar_ast)
        dap_sum = paired_wilcoxon_summary(p_dap_raw, p_dap_ast)
        sign_sum = compute_exact_sign_test(p_ochiai_raw, p_ochiai_ast)

        n_values = [1, 3, 5, 10]
        acc_sum: Dict[str, Any] = {}
        if b_ranks:
            acc_sum = {
                f"acc@{n}": {
                    "baseline": sum(1 for rk in b_ranks if rk <= n),
                    "raw_diff": sum(1 for rk in r_ranks if rk <= n),
                    "ast_edit": sum(1 for rk in a_ranks if rk <= n),
                    "delta_raw": sum(1 for rk in r_ranks if rk <= n) - sum(1 for rk in b_ranks if rk <= n),
                    "delta_ast": sum(1 for rk in a_ranks if rk <= n) - sum(1 for rk in b_ranks if rk <= n),
                }
                for n in n_values
            }

        return {
            "paired_bugs": paired_bugs,
            "ochiai_raw_vals": p_ochiai_raw,
            "ochiai_ast_vals": p_ochiai_ast,
            "dstar_raw_vals": p_dstar_raw,
            "dstar_ast_vals": p_dstar_ast,
            "ochiai": ochiai_sum,
            "dstar": dstar_sum,
            "delta_ap": dap_sum,
            "sign_test": sign_sum,
            "acc_at_n": acc_sum,
        }

    # Primary RQ2: strict assertion-only paired subset; fallback to LIBRO only if strict has 0 spectra
    paired_strict = _collect_paired_fdg(strict_only=True)
    paired_libro = _collect_paired_fdg(strict_only=False)
    primary_paired = paired_strict if paired_strict["ochiai"]["n_paired"] > 0 else paired_libro

    rq2_ochiai = primary_paired["ochiai"]
    rq2_dstar = primary_paired["dstar"]
    rq2_delta_ap = primary_paired["delta_ap"]
    sign_test_ochiai = primary_paired["sign_test"]
    acc_summary = primary_paired["acc_at_n"]

    # Global / Intention-To-Treat Expected FDG (E[FDG]) across all evaluated bugs
    expected_fdg_raw_strict, expected_fdg_ast_strict = [], []
    expected_fdg_raw_libro, expected_fdg_ast_libro = [], []
    for r in eval_records:
        fdg = r.get("fdg", {})
        is_raw_s = bool(r["verdicts"].get("raw_diff", {}).get("is_valid_strict_assertion", False))
        is_ast_s = bool(r["verdicts"].get("ast_edit", {}).get("is_valid_strict_assertion", False))
        is_raw_l = bool(r["verdicts"].get("raw_diff", {}).get("is_valid_libro", False))
        is_ast_l = bool(r["verdicts"].get("ast_edit", {}).get("is_valid_libro", False))
        raw_o = float(fdg.get("raw_diff", {}).get("ochiai", {}).get("fdg", 0.0))
        ast_o = float(fdg.get("ast_edit", {}).get("ochiai", {}).get("fdg", 0.0))
        expected_fdg_raw_strict.append(raw_o if is_raw_s else 0.0)
        expected_fdg_ast_strict.append(ast_o if is_ast_s else 0.0)
        expected_fdg_raw_libro.append(raw_o if is_raw_l else 0.0)
        expected_fdg_ast_libro.append(ast_o if is_ast_l else 0.0)

    rq2_expected_fdg_strict = paired_wilcoxon_summary(expected_fdg_raw_strict, expected_fdg_ast_strict)
    rq2_expected_fdg_libro = paired_wilcoxon_summary(expected_fdg_raw_libro, expected_fdg_ast_libro)

    # Project-stratified RQ1 breakdown (both Strict primary and LIBRO secondary)
    rq1_by_project_strict: Dict[str, Any] = {}
    rq1_by_project: Dict[str, Any] = {}
    for proj in sorted({str(r.get("project", "Unknown")) for r in eval_records}):
        sub = [r for r in eval_records if str(r.get("project", "Unknown")) == proj]
        prs = [bool(r["verdicts"].get("raw_diff", {}).get("is_valid_strict_assertion", False)) for r in sub]
        pas = [bool(r["verdicts"].get("ast_edit", {}).get("is_valid_strict_assertion", False)) for r in sub]
        prl = [bool(r["verdicts"].get("raw_diff", {}).get("is_valid_libro", False)) for r in sub]
        pal = [bool(r["verdicts"].get("ast_edit", {}).get("is_valid_libro", False)) for r in sub]
        rq1_by_project_strict[proj] = mcnemar_exact(prs, pas)
        rq1_by_project[proj] = mcnemar_exact(prl, pal)

    rq1_failure_modes: Dict[str, Any] = {}
    for cond in ("raw_diff", "ast_edit"):
        cls_counts: Dict[str, int] = {}
        for r in eval_records:
            c = str(r["verdicts"].get(cond, {}).get("classification", "MISSING"))
            cls_counts[c] = cls_counts.get(c, 0) + 1
        n_fail = sum(v for k, v in cls_counts.items() if not k.startswith("VALID_"))
        rq1_failure_modes[cond] = {
            "counts": cls_counts,
            "total_failures": n_fail,
            "fail_fail_share_of_failures": round(cls_counts.get("FAIL_FAIL", 0) / max(1, n_fail), 4),
            "prefix_pass_share_of_failures": round(cls_counts.get("PREFIX_PASS", 0) / max(1, n_fail), 4),
        }

    # Compute RQ3 efficiency metrics directly from trial logs when available
    rq3_efficiency: Dict[str, Any] = {}
    if trials_dir is not None and trials_dir.exists():
        for cond in ("raw_diff", "ast_edit"):
            toks, atts, ctxs, walls = [], [], [], []
            for r in eval_records:
                bk = r.get("bug_key", "")
                tp = trials_dir / f"{bk}__{cond}.json"
                if not tp.exists():
                    continue
                tr = json.loads(tp.read_text(encoding="utf-8"))
                toks.append(int(tr.get("prompt_tokens", 0)) + int(tr.get("completion_tokens", 0)))
                atts.append(int(tr.get("num_run_test_attempts", 0)))
                ctxs.append(int(tr.get("num_get_context_calls", 0)))
                walls.append(float(tr.get("wall_time_sec", 0.0)))
            if toks:
                rq3_efficiency[cond] = {
                    "n_trials": len(toks),
                    "total_tokens": sum(toks),
                    "mean_tokens": round(sum(toks) / len(toks), 1),
                    "mean_attempts": round(sum(atts) / len(atts), 2),
                    "mean_get_context_calls": round(sum(ctxs) / len(ctxs), 2),
                    "trials_with_get_context": sum(1 for c in ctxs if c > 0),
                    "mean_wall_time_sec": round(sum(walls) / len(walls), 2),
                }
        if "raw_diff" in rq3_efficiency and "ast_edit" in rq3_efficiency:
            r_tok = rq3_efficiency["raw_diff"]["mean_tokens"]
            a_tok = rq3_efficiency["ast_edit"]["mean_tokens"]
            if r_tok > 0:
                rq3_efficiency["token_reduction_pct"] = round((r_tok - a_tok) / r_tok * 100.0, 1)

    summary = {
        "rq1_strict_assertion_validity": rq1_strict,
        "rq1_libro_validity": rq1_libro,
        "rq1_by_project_strict": rq1_by_project_strict,
        "rq1_by_project": rq1_by_project,
        "rq1_failure_modes": rq1_failure_modes,
        "rq2_fdg_ochiai_paired": rq2_ochiai,
        "rq2_fdg_dstar_paired": rq2_dstar,
        "rq2_delta_ap_ochiai_paired": rq2_delta_ap,
        "rq2_sign_test_ochiai": sign_test_ochiai,
        "rq2_acc_at_n": acc_summary,
        "rq2_paired_bugs_strict": paired_strict["paired_bugs"],
        "rq2_fdg_ochiai_paired_libro": paired_libro["ochiai"],
        "rq2_fdg_dstar_paired_libro": paired_libro["dstar"],
        "rq2_delta_ap_ochiai_paired_libro": paired_libro["delta_ap"],
        "rq2_sign_test_ochiai_libro": paired_libro["sign_test"],
        "rq2_acc_at_n_libro": paired_libro["acc_at_n"],
        "rq2_expected_fdg_global": rq2_expected_fdg_strict,
        "rq2_expected_fdg_global_libro": rq2_expected_fdg_libro,
        "rq3_efficiency": rq3_efficiency,
    }

    # Write LaTeX Table for RQ1 (Strict Assertion-Only PRIMARY, LIBRO secondary)
    latex_rq1 = rf"""\begin{{table}}[t]
\centering
\caption{{RQ1: Bug-Reproducing Test Validity --- Raw Diff (Control) vs. AST Edit Script (Treatment), $N={rq1_strict['n_total']}$.}}
\label{{tab:rq1_validity}}
\begin{{tabular}}{{lrrrr}}
\toprule
Validity Criterion & Raw Diff & AST Edit Script & Both Valid & McNemar $p$ \\
\midrule
Strict Assertion-Only (Primary) & {rq1_strict['raw_rate']*100:.1f}\% & {rq1_strict['ast_rate']*100:.1f}\% & {rq1_strict['both_valid']} & {rq1_strict['p_value']:.4f} \\
LIBRO Fail/Pass (Secondary) & {rq1_libro['raw_rate']*100:.1f}\% & {rq1_libro['ast_rate']*100:.1f}\% & {rq1_libro['both_valid']} & {rq1_libro['p_value']:.4f} \\
\bottomrule
\end{{tabular}}
\end{{table}}
"""
    (figures_dir / "table_rq1.tex").write_text(latex_rq1, encoding="utf-8")

    # Write LaTeX Table for RQ2 (with simple paired counts, Wilcoxon, A12)
    paired_counts_str = (
        f"{sign_test_ochiai['n_positive_ast']}/{sign_test_ochiai['n_exact_ties']}/{sign_test_ochiai['n_negative_raw']}"
    )
    latex_rq2 = rf"""\begin{{table}}[t]
\centering
\small
\caption{{RQ2: Paired Fault Diagnosability Gain (FDG, $\alpha=0.5$) and $\Delta$AP over Strict-Valid Paired Subset ($N_{{\text{{paired}}}}={rq2_ochiai['n_paired']}$; AST wins/ties/raw wins = {paired_counts_str}).}}
\label{{tab:rq2_fdg}}
\resizebox{{\columnwidth}}{{!}}{{%
\begin{{tabular}}{{lrrrrr}}
\toprule
Metric & Raw Diff Mean & AST Edit Mean & $\Delta$ (AST$-$Raw) & Wilcoxon $p$ & $\hat{{A}}_{{12}}$ \\
\midrule
FDG (Ochiai, primary) & {rq2_ochiai.get('raw_mean', 0):.4f} & {rq2_ochiai.get('ast_mean', 0):.4f} & {rq2_ochiai.get('mean_diff_ast_minus_raw', 0):+.4f} & {rq2_ochiai['p_value']:.4f} & {rq2_ochiai['a12']:.3f} \\
FDG (DStar, $*=2$) & {rq2_dstar.get('raw_mean', 0):.4f} & {rq2_dstar.get('ast_mean', 0):.4f} & {rq2_dstar.get('mean_diff_ast_minus_raw', 0):+.4f} & {rq2_dstar['p_value']:.4f} & {rq2_dstar['a12']:.3f} \\
$\Delta$AP (Ochiai) & {rq2_delta_ap.get('raw_mean', 0):+.4f} & {rq2_delta_ap.get('ast_mean', 0):+.4f} & {rq2_delta_ap.get('mean_diff_ast_minus_raw', 0):+.4f} & {rq2_delta_ap['p_value']:.4f} & {rq2_delta_ap['a12']:.3f} \\
\bottomrule
\end{{tabular}}%
}}
\end{{table}}
"""
    (figures_dir / "table_rq2.tex").write_text(latex_rq2, encoding="utf-8")

    n_total = rq1_strict["n_total"]
    n_paired = rq2_ochiai["n_paired"]
    raw_strict_count = rq1_strict["both_valid"] + rq1_strict["raw_only_valid"]
    ast_strict_count = rq1_strict["both_valid"] + rq1_strict["ast_only_valid"]
    raw_libro_count = rq1_libro["both_valid"] + rq1_libro["raw_only_valid"]
    ast_libro_count = rq1_libro["both_valid"] + rq1_libro["ast_only_valid"]

    raw_tok_str = f"{rq3_efficiency.get('raw_diff', {}).get('mean_tokens', 0):,.0f}" if rq3_efficiency else "N/A"
    ast_tok_str = f"{rq3_efficiency.get('ast_edit', {}).get('mean_tokens', 0):,.0f}" if rq3_efficiency else "N/A"
    tok_red_str = f"$-{rq3_efficiency.get('token_reduction_pct', 0):.1f}\\%$ for AST" if "token_reduction_pct" in rq3_efficiency else "---"
    raw_att_str = f"{rq3_efficiency.get('raw_diff', {}).get('mean_attempts', 0):.2f}" if rq3_efficiency else "N/A"
    ast_att_str = f"{rq3_efficiency.get('ast_edit', {}).get('mean_attempts', 0):.2f}" if rq3_efficiency else "N/A"

    latex_master = rf"""\begin{{table}}[t]
\centering
\small
\caption{{Master Experimental Summary: Raw Unified Diff (Control) vs. Normalized AST Edit Script (Treatment) across all dimensions ($N={n_total}$ Defects4J Bugs, {n_total * 2} Multi-Turn Trials).}}
\label{{tab:summary_master}}
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{llrrc}}
\toprule
Evaluation Dimension & Metric / Scope & Raw Diff & AST Edit & Statistical Test / Counts \\
\midrule
\multirow{{2}}{{*}}{{\textbf{{RQ1: BRT Validity}}}} & \textbf{{Strict Assertion-Only (Primary)}}, $N={n_total}$ & \textbf{{{rq1_strict['raw_rate']*100:.1f}\%}} ({raw_strict_count}/{n_total}) & {rq1_strict['ast_rate']*100:.1f}\% ({ast_strict_count}/{n_total}) & McNemar $p = {rq1_strict['p_value']:.4f}$ \\
 & LIBRO Fail/Pass (Secondary), $N={n_total}$ & \textbf{{{rq1_libro['raw_rate']*100:.1f}\%}} ({raw_libro_count}/{n_total}) & {rq1_libro['ast_rate']*100:.1f}\% ({ast_libro_count}/{n_total}) & McNemar $p = {rq1_libro['p_value']:.4f}$ \\
\midrule
\multirow{{4}}{{*}}{{\textbf{{RQ2: Diagnostic Quality}}}} & Ochiai FDG (Strict Paired, $N={n_paired}$) & {rq2_ochiai.get('raw_mean', 0):.4f} & {rq2_ochiai.get('ast_mean', 0):.4f} & Wilcoxon $p = {rq2_ochiai['p_value']:.4f}$, counts {paired_counts_str} \\
 & DStar FDG (Strict Paired, $N={n_paired}$) & {rq2_dstar.get('raw_mean', 0):.4f} & {rq2_dstar.get('ast_mean', 0):.4f} & Wilcoxon $p = {rq2_dstar['p_value']:.4f}$ ($\hat{{A}}_{{12}} = {rq2_dstar['a12']:.3f}$) \\
 & Realized $\Delta$AP (Ochiai, $N={n_paired}$) & {rq2_delta_ap.get('raw_mean', 0):+.4f} & {rq2_delta_ap.get('ast_mean', 0):+.4f} & Wilcoxon $p = {rq2_delta_ap['p_value']:.4f}$ ($\hat{{A}}_{{12}} = {rq2_delta_ap['a12']:.3f}$) \\
 & \textbf{{Expected Diagnosability $E[\text{{FDG}}]$ (Strict, $N={n_total}$)}} & \textbf{{{rq2_expected_fdg_strict.get('raw_mean', 0):.4f}}} & {rq2_expected_fdg_strict.get('ast_mean', 0):.4f} & Wilcoxon $p = {rq2_expected_fdg_strict['p_value']:.4f}$ ($\hat{{A}}_{{12}} = {rq2_expected_fdg_strict['a12']:.3f}$) \\
\midrule
\multirow{{2}}{{*}}{{\textbf{{RQ3: Efficiency \& Iterations}}}} & Mean Token Overhead / Bug & {raw_tok_str} & \textbf{{{ast_tok_str}}} & {tok_red_str} \\
 & Mean Iteration Attempts / Bug & {raw_att_str} & {ast_att_str} & 5-retry budget \\
\bottomrule
\end{{tabular}}%
}}
\end{{table}}
"""
    (figures_dir / "table_summary_metrics.tex").write_text(latex_master, encoding="utf-8")

    # Plot 1: RQ1 Validity Rate Comparison (Strict Assertion-Only FIRST, LIBRO SECOND)
    fig, ax = plt.subplots(figsize=(6.2, 4.0), dpi=300)
    labels = ["Strict Assertion-Only\n(Primary)", "LIBRO Fail/Pass\n(Secondary)"]
    raw_pcts = [rq1_strict["raw_rate"] * 100, rq1_libro["raw_rate"] * 100]
    ast_pcts = [rq1_strict["ast_rate"] * 100, rq1_libro["ast_rate"] * 100]
    x = np.arange(len(labels))
    width = 0.34
    ax.bar(x - width / 2, raw_pcts, width, label="Raw Diff (Control)", color="#4E79A7")
    ax.bar(x + width / 2, ast_pcts, width, label="AST Edit Script (Treatment)", color="#F28E2B")
    ax.set_ylabel("Valid BRT Rate (%)")
    ax.set_title("RQ1: Bug-Reproducing Test Validity Rate")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figures_dir / "rq1_validity_comparison.png", dpi=300)
    plt.close(fig)

    # Plot 2: RQ2 Paired FDG Boxplot (Strict-Valid Paired Subset)
    p_ochiai_raw_plot = primary_paired["ochiai_raw_vals"]
    p_ochiai_ast_plot = primary_paired["ochiai_ast_vals"]
    p_dstar_raw_plot = primary_paired["dstar_raw_vals"]
    p_dstar_ast_plot = primary_paired["dstar_ast_vals"]
    if p_ochiai_raw_plot:
        fig, axes = plt.subplots(1, 2, figsize=(8.2, 4.0), dpi=300, sharey=True)
        axes[0].boxplot(
            [p_ochiai_raw_plot, p_ochiai_ast_plot],
            patch_artist=True,
        )
        axes[0].set_xticks([1, 2])
        axes[0].set_xticklabels(["Raw Diff", "AST Edit"])
        axes[0].set_title(f"FDG (Ochiai, N={len(p_ochiai_raw_plot)})")
        axes[0].set_ylabel("FDG Score (alpha = 0.5)")
        axes[0].grid(axis="y", linestyle="--", alpha=0.4)

        axes[1].boxplot(
            [p_dstar_raw_plot, p_dstar_ast_plot],
            patch_artist=True,
        )
        axes[1].set_xticks([1, 2])
        axes[1].set_xticklabels(["Raw Diff", "AST Edit"])
        axes[1].set_title(f"FDG (DStar *=2, N={len(p_dstar_raw_plot)})")
        axes[1].grid(axis="y", linestyle="--", alpha=0.4)
        fig.tight_layout()
        fig.savefig(figures_dir / "rq2_fdg_paired.png", dpi=300)
        plt.close(fig)

    (figures_dir / "statistical_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    swe_log_path = Path("data/swe_bench_exclusion_log.json")
    if swe_log_path.exists():
        swe = json.loads(swe_log_path.read_text(encoding="utf-8"))
        ec = swe.get("exclusion_counts", {})
        latex_mswe = rf"""\begin{{table}}[t]
\centering
\caption{{Multi-SWE-bench (Java) Curation Audit with Explicit Five-Stage Exclusion Criteria.}}
\label{{tab:curation_multiswe}}
\begin{{tabular}}{{lr}}
\toprule
Screening Stage / Exclusion Filter & PR Count \\
\midrule
Total Java PRs scanned (across 9 repositories) & {swe.get('total_scanned', 0)} \\
Excluded: Issue/PR leaked JUnit test code (Filter 1) & {ec.get('issue_leaked_brt', 0)} \\
Excluded: Touched non-Java/test files or $\neq 1$ file (Filter 2) & {ec.get('non_single_java_file', 0)} \\
Excluded: Multi-method or class-level edit (Filter 3) & {ec.get('multi_method_fix', 0)} \\
Excluded: Base checkout / patch failure (Filter 4) & {ec.get('clone_or_patch_failed', 0)} \\
Excluded: GumTree AST parse failure (Filter 5) & {ec.get('gumtree_parse_failed', 0)} \\
\midrule
\textbf{{Curated Single-Method Java PRs (7 repositories)}} & \textbf{{{swe.get('total_curated', 0)}}} \\
\bottomrule
\end{{tabular}}
\end{{table}}
"""
        (figures_dir / "table_curation_multiswe.tex").write_text(latex_mswe, encoding="utf-8")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate statistical tests, LaTeX tables, and plots.")
    parser.add_argument("--config", default="configs/experiment.yaml")
    args = parser.parse_args()
    cfg = ExperimentConfig.from_yaml(args.config)
    eval_path = cfg.paths.results_dir / "evaluation_summary.json"
    records = json.loads(eval_path.read_text(encoding="utf-8"))
    trials_dir = cfg.paths.results_dir / "trials"
    summary = analyze_evaluation_records(records, cfg.paths.figures_dir, trials_dir=trials_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
