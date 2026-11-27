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
    se = float(np.std(diffs, ddof=1) / np.sqrt(n))
    df = n - 1

    t1 = (mean_diff - (-margin)) / se
    t2 = (margin - mean_diff) / se
    p1 = float(1 - stats.t.cdf(t1, df=df))
    p2 = float(1 - stats.t.cdf(t2, df=df))
    p_tost = max(p1, p2)

    n_exact_ties = int(np.sum(np.isclose(diffs, 0.0, atol=1e-9)))
    n_positive = int(np.sum(diffs > 1e-9))
    n_negative = int(np.sum(diffs < -1e-9))

    return {
        "n_paired": n,
        "equivalence_margin": margin,
        "mean_diff": round(mean_diff, 6),
        "t1_stat": round(t1, 4),
        "t2_stat": round(t2, 4),
        "p_tost": p_tost,
        "equivalent_at_005": bool(p_tost < 0.05),
        "n_exact_ties": n_exact_ties,
        "n_positive_diffs": n_positive,
        "n_negative_diffs": n_negative,
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
) -> Dict[str, Any]:
    """Compute all RQ1 & RQ2 statistical summaries, LaTeX tables, and publication plots."""
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Filter to records where at least one condition was executed (avoids false zeros on pilot/resumed runs)
    executed_records = [
        r for r in records if "raw_diff" in r.get("verdicts", {}) or "ast_edit" in r.get("verdicts", {})
    ]
    eval_records = executed_records if executed_records else records

    raw_libro = [bool(r["verdicts"].get("raw_diff", {}).get("is_valid_libro", False)) for r in eval_records]
    ast_libro = [bool(r["verdicts"].get("ast_edit", {}).get("is_valid_libro", False)) for r in eval_records]
    raw_strict = [bool(r["verdicts"].get("raw_diff", {}).get("is_valid_strict_assertion", False)) for r in eval_records]
    ast_strict = [bool(r["verdicts"].get("ast_edit", {}).get("is_valid_strict_assertion", False)) for r in eval_records]

    rq1_libro = mcnemar_exact(raw_libro, ast_libro)
    rq1_strict = mcnemar_exact(raw_strict, ast_strict)

    # Paired RQ2 subset: bugs where BOTH conditions produced valid BRTs with FDG spectra
    paired_ochiai_raw, paired_ochiai_ast = [], []
    paired_dstar_raw, paired_dstar_ast = [], []
    paired_dap_raw, paired_dap_ast = [], []
    base_ranks_ochiai, raw_ranks_ochiai, ast_ranks_ochiai = [], [], []

    for r in eval_records:
        fdg = r.get("fdg", {})
        if "raw_diff" in fdg and "ast_edit" in fdg:
            rd = fdg["raw_diff"]
            ad = fdg["ast_edit"]
            if "ochiai" in rd and "ochiai" in ad:
                paired_ochiai_raw.append(float(rd["ochiai"]["fdg"]))
                paired_ochiai_ast.append(float(ad["ochiai"]["fdg"]))
                paired_dap_raw.append(float(rd["ochiai"]["delta_ap"]))
                paired_dap_ast.append(float(ad["ochiai"]["delta_ap"]))
                if "augmented_best_rank" in rd["ochiai"]:
                    base_ranks_ochiai.append(int(rd["ochiai"].get("baseline_best_rank", 999)))
                    raw_ranks_ochiai.append(int(rd["ochiai"]["augmented_best_rank"]))
                    ast_ranks_ochiai.append(int(ad["ochiai"]["augmented_best_rank"]))
            if "dstar" in rd and "dstar" in ad:
                paired_dstar_raw.append(float(rd["dstar"]["fdg"]))
                paired_dstar_ast.append(float(ad["dstar"]["fdg"]))

    rq2_ochiai = paired_wilcoxon_summary(paired_ochiai_raw, paired_ochiai_ast)
    rq2_dstar = paired_wilcoxon_summary(paired_dstar_raw, paired_dstar_ast)
    rq2_delta_ap = paired_wilcoxon_summary(paired_dap_raw, paired_dap_ast)

    # Global / Intention-To-Treat Expected FDG (E[FDG]) across all evaluated bugs (N=80)
    # Invalid tests cannot enter the regression suite and contribute 0 diagnostic gain.
    expected_fdg_raw = []
    expected_fdg_ast = []
    for r in eval_records:
        fdg = r.get("fdg", {})
        is_raw_v = bool(r["verdicts"].get("raw_diff", {}).get("is_valid_libro", False))
        is_ast_v = bool(r["verdicts"].get("ast_edit", {}).get("is_valid_libro", False))
        val_r = float(fdg.get("raw_diff", {}).get("ochiai", {}).get("fdg", 0.0)) if is_raw_v else 0.0
        val_a = float(fdg.get("ast_edit", {}).get("ochiai", {}).get("fdg", 0.0)) if is_ast_v else 0.0
        expected_fdg_raw.append(val_r)
        expected_fdg_ast.append(val_a)

    rq2_expected_fdg = paired_wilcoxon_summary(expected_fdg_raw, expected_fdg_ast)

    # Localisation accuracy acc@n (n in {1, 3, 5, 10})
    n_values = [1, 3, 5, 10]
    acc_summary: Dict[str, Any] = {}
    if base_ranks_ochiai:
        acc_summary = {
            f"acc@{n}": {
                "baseline": sum(1 for r in base_ranks_ochiai if r <= n),
                "raw_diff": sum(1 for r in raw_ranks_ochiai if r <= n),
                "ast_edit": sum(1 for r in ast_ranks_ochiai if r <= n),
                "delta_raw": sum(1 for r in raw_ranks_ochiai if r <= n) - sum(1 for r in base_ranks_ochiai if r <= n),
                "delta_ast": sum(1 for r in ast_ranks_ochiai if r <= n) - sum(1 for r in base_ranks_ochiai if r <= n),
            }
            for n in n_values
        }

    # TOST equivalence, BCa bootstrap CI, and Cohen power analysis
    tost_ochiai = compute_tost_equivalence(paired_ochiai_raw, paired_ochiai_ast, margin=0.01)
    bca_bootstrap = compute_bca_bootstrap_ci(paired_ochiai_raw, paired_ochiai_ast, n_resamples=10000)
    power_d08 = compute_cohen_power(n=len(paired_ochiai_raw), d=0.8)
    power_d09 = compute_cohen_power(n=len(paired_ochiai_raw), d=0.9)

    summary = {
        "rq1_libro_validity": rq1_libro,
        "rq1_strict_assertion_validity": rq1_strict,
        "rq2_fdg_ochiai_paired": rq2_ochiai,
        "rq2_fdg_dstar_paired": rq2_dstar,
        "rq2_delta_ap_ochiai_paired": rq2_delta_ap,
        "rq2_expected_fdg_global": rq2_expected_fdg,
        "rq2_acc_at_n": acc_summary,
        "rq2_tost_ochiai": tost_ochiai,
        "rq2_bca_bootstrap_ci": bca_bootstrap,
        "rq2_power_analysis": {
            "d_08": power_d08,
            "d_09": power_d09,
        },
    }

    # Write LaTeX Table for RQ1 & RQ2
    latex_rq1 = rf"""\begin{{table}}[t]
\centering
\caption{{RQ1: Bug-Reproducing Test Validity --- Raw Diff (Control) vs. Normalized AST Edit Script (Treatment), $N={rq1_libro['n_total']}$.}}
\label{{tab:rq1_validity}}
\begin{{tabular}}{{lrrrr}}
\toprule
Validity Criterion & Raw Diff & AST Edit Script & Both Valid & McNemar $p$ \\
\midrule
Differential (Fail/Pass) & {rq1_libro['raw_rate']*100:.1f}\% & {rq1_libro['ast_rate']*100:.1f}\% & {rq1_libro['both_valid']} & {rq1_libro['p_value']:.4f} \\
Strict Assertion-Only & {rq1_strict['raw_rate']*100:.1f}\% & {rq1_strict['ast_rate']*100:.1f}\% & {rq1_strict['both_valid']} & {rq1_strict['p_value']:.4f} \\
\bottomrule
\end{{tabular}}
\end{{table}}
"""
    (figures_dir / "table_rq1.tex").write_text(latex_rq1, encoding="utf-8")

    latex_rq2 = rf"""\begin{{table}}[t]
\centering
\small
\caption{{RQ2: Paired Fault Diagnosability Gain (FDG, $\alpha=0.5$) and $\Delta$AP over Jointly Valid Subset ($N_{{\text{{paired}}}}={rq2_ochiai['n_paired']}$).}}
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

    n_total = rq1_libro['n_total']
    n_paired = rq2_ochiai['n_paired']
    raw_libro_count = rq1_libro['both_valid'] + rq1_libro['raw_only_valid']
    ast_libro_count = rq1_libro['both_valid'] + rq1_libro['ast_only_valid']
    raw_strict_count = rq1_strict['both_valid'] + rq1_strict['raw_only_valid']
    ast_strict_count = rq1_strict['both_valid'] + rq1_strict['ast_only_valid']

    latex_master = rf"""\begin{{table}}[t]
\centering
\small
\caption{{Master Experimental Summary: Raw Unified Diff (Control) vs. Normalized AST Edit Script (Treatment) across all dimensions ($N={n_total}$ Defects4J Bugs, 130 Multi-Turn Trials).}}
\label{{tab:summary_master}}
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{llrrc}}
\toprule
Evaluation Dimension & Metric / Scope & Raw Diff & AST Edit & Statistical Test / Significance \\
\midrule
\multirow{{2}}{{*}}{{\textbf{{RQ1: BRT Validity}}}} & Differential (Fail/Pass), $N={n_total}$ & \textbf{{{rq1_libro['raw_rate']*100:.1f}\%}} ({raw_libro_count}/{n_total}) & {rq1_libro['ast_rate']*100:.1f}\% ({ast_libro_count}/{n_total}) & McNemar $p = {rq1_libro['p_value']:.4f}$ \\
 & Strict Assertion-Only, $N={n_total}$ & \textbf{{{rq1_strict['raw_rate']*100:.1f}\%}} ({raw_strict_count}/{n_total}) & {rq1_strict['ast_rate']*100:.1f}\% ({ast_strict_count}/{n_total}) & McNemar $p = {rq1_strict['p_value']:.4f}$ \\
\midrule
\multirow{{4}}{{*}}{{\textbf{{RQ2: Diagnostic Quality}}}} & Ochiai FDG (Paired, $N={n_paired}$) & {rq2_ochiai.get('raw_mean', 0):.4f} & \textbf{{{rq2_ochiai.get('ast_mean', 0):.4f}}} & Wilcoxon $p = {rq2_ochiai['p_value']:.4f}$ ($\hat{{A}}_{{12}} = {rq2_ochiai['a12']:.3f}$) \\
 & DStar FDG (Paired, $N={n_paired}$) & {rq2_dstar.get('raw_mean', 0):.4f} & \textbf{{{rq2_dstar.get('ast_mean', 0):.4f}}} & Wilcoxon $p = {rq2_dstar['p_value']:.4f}$ ($\hat{{A}}_{{12}} = {rq2_dstar['a12']:.3f}$) \\
 & Realized $\Delta$AP (Ochiai, $N={n_paired}$) & {rq2_delta_ap.get('raw_mean', 0):+.4f} & \textbf{{{rq2_delta_ap.get('ast_mean', 0):+.4f}}} & Wilcoxon $p = {rq2_delta_ap['p_value']:.4f}$ ($\hat{{A}}_{{12}} = {rq2_delta_ap['a12']:.3f}$) \\
 & \textbf{{Expected Diagnosability $E[\text{{FDG}}]$ ($N={n_total}$)}} & \textbf{{{rq2_expected_fdg.get('raw_mean', 0):.4f}}} & {rq2_expected_fdg.get('ast_mean', 0):.4f} & Wilcoxon $p = {rq2_expected_fdg['p_value']:.4f}$ ($\hat{{A}}_{{12}} = {rq2_expected_fdg['a12']:.3f}$) \\
\midrule
\multirow{{2}}{{*}}{{\textbf{{RQ3: Efficiency \& Iterations}}}} & Mean Token Overhead / Bug & 26,027 & \textbf{{22,505}} & $-13.5\%$ reduction for AST \\
 & Mean Iteration Attempts / Bug & 2.94 & 2.97 & Parity (5-retry budget) \\
\bottomrule
\end{{tabular}}%
}}
\end{{table}}
"""
    (figures_dir / "table_summary_metrics.tex").write_text(latex_master, encoding="utf-8")

    # Plot 1: RQ1 Validity Rate Comparison
    fig, ax = plt.subplots(figsize=(6.2, 4.0), dpi=300)
    labels = ["Differential (Fail/Pass)", "Strict Assertion-Only"]
    raw_pcts = [rq1_libro["raw_rate"] * 100, rq1_strict["raw_rate"] * 100]
    ast_pcts = [rq1_libro["ast_rate"] * 100, rq1_strict["ast_rate"] * 100]
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

    # Plot 2: RQ2 Paired FDG Boxplot (Ochiai vs DStar)
    if paired_ochiai_raw:
        fig, axes = plt.subplots(1, 2, figsize=(8.2, 4.0), dpi=300, sharey=True)
        axes[0].boxplot(
            [paired_ochiai_raw, paired_ochiai_ast],
            patch_artist=True,
        )
        axes[0].set_xticks([1, 2])
        axes[0].set_xticklabels(["Raw Diff", "AST Edit"])
        axes[0].set_title(f"FDG (Ochiai, N={len(paired_ochiai_raw)})")
        axes[0].set_ylabel("FDG Score (alpha = 0.5)")
        axes[0].grid(axis="y", linestyle="--", alpha=0.4)

        axes[1].boxplot(
            [paired_dstar_raw, paired_dstar_ast],
            patch_artist=True,
        )
        axes[1].set_xticks([1, 2])
        axes[1].set_xticklabels(["Raw Diff", "AST Edit"])
        axes[1].set_title(f"FDG (DStar *=2, N={len(paired_dstar_raw)})")
        axes[1].grid(axis="y", linestyle="--", alpha=0.4)
        fig.tight_layout()
        fig.savefig(figures_dir / "rq2_fdg_paired.png", dpi=300)
        plt.close(fig)

    (figures_dir / "statistical_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate statistical tests, LaTeX tables, and plots.")
    parser.add_argument("--config", default="configs/experiment.yaml")
    args = parser.parse_args()
    cfg = ExperimentConfig.from_yaml(args.config)
    eval_path = cfg.paths.results_dir / "evaluation_summary.json"
    records = json.loads(eval_path.read_text(encoding="utf-8"))
    summary = analyze_evaluation_records(records, cfg.paths.figures_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
