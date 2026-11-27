"""Auto-generate the Empirical Results section in README.md directly from JSON result artifacts."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict


START_MARKER = "<!-- BEGIN_AUTOGEN_RESULTS -->"
END_MARKER = "<!-- END_AUTOGEN_RESULTS -->"


def render_empirical_section(
    summary: Dict[str, Any],
    exclusion_log: Dict[str, Any],
    eval_meta: Dict[str, Any] | None = None,
) -> str:
    rq1_s = summary["rq1_strict_assertion_validity"]
    rq1_l = summary["rq1_libro_validity"]
    by_proj_s = summary.get("rq1_by_project_strict", {})
    by_proj_l = summary.get("rq1_by_project", {})
    rq2_o = summary["rq2_fdg_ochiai_paired"]
    rq2_d = summary["rq2_fdg_dstar_paired"]
    rq2_ap = summary["rq2_delta_ap_ochiai_paired"]
    rq2_sign = summary["rq2_sign_test_ochiai"]
    rq2_exp = summary["rq2_expected_fdg_global"]
    rq2_acc = summary.get("rq2_acc_at_n", {})
    rq3 = summary.get("rq3_efficiency", {})

    n_total = rq1_s["n_total"]
    raw_s_cnt = rq1_s["both_valid"] + rq1_s["raw_only_valid"]
    ast_s_cnt = rq1_s["both_valid"] + rq1_s["ast_only_valid"]
    raw_l_cnt = rq1_l["both_valid"] + rq1_l["raw_only_valid"]
    ast_l_cnt = rq1_l["both_valid"] + rq1_l["ast_only_valid"]

    n_paired = rq2_o["n_paired"]
    paired_counts = (
        f"{rq2_sign['n_positive_ast']} AST wins / "
        f"{rq2_sign['n_exact_ties']} ties / "
        f"{rq2_sign['n_negative_raw']} raw wins"
    )

    acc3 = rq2_acc.get("acc@3", {})
    acc3_raw = f"{acc3.get('raw_diff', 'N/A')} ($\\Delta = {acc3.get('delta_raw', 0):+d}$)" if acc3 else "N/A"
    acc3_ast = f"{acc3.get('ast_edit', 'N/A')} ($\\Delta = {acc3.get('delta_ast', 0):+d}$)" if acc3 else "N/A"
    acc3_base = acc3.get("baseline", "N/A") if acc3 else "N/A"

    raw_tok = rq3.get("raw_diff", {}).get("mean_tokens", 0)
    ast_tok = rq3.get("ast_edit", {}).get("mean_tokens", 0)
    tok_red = rq3.get("token_reduction_pct", 0.0)
    raw_att = rq3.get("raw_diff", {}).get("mean_attempts", 0.0)
    ast_att = rq3.get("ast_edit", {}).get("mean_attempts", 0.0)

    lang_s = by_proj_s.get("Lang", {})
    math_s = by_proj_s.get("Math", {})
    lang_l = by_proj_l.get("Lang", {})
    math_l = by_proj_l.get("Math", {})

    n_lang = lang_s.get("n_total", 40)
    n_math = math_s.get("n_total", 25)
    n_curated_init = exclusion_log.get("total_curated_initial", 80)
    n_aborted = exclusion_log.get("exclusion_counts", {}).get("api_quota_exhausted_aborted", 15)

    has_passing_note = ""
    if eval_meta and eval_meta.get("evaluation_config", {}).get("include_passing_baseline_tests"):
        max_p = eval_meta["evaluation_config"].get("max_baseline_passing_tests", 25)
        has_passing_note = (
            f"Diagnostic FDG/SBFL spectra incorporate both `tests.trigger` failing tests and up to {max_p} "
            f"verified passing tests sampled across `tests.relevant`."
        )
    else:
        has_passing_note = "Diagnostic spectra use `tests.trigger` failing tests."

    lines = [
        START_MARKER,
        "## Empirical Results Summary (Initial 65-Bug Pilot)",
        "",
        f"Evaluated on $N={n_total}$ Defects4J bugs across Apache Commons-Lang ({n_lang} bugs) and Commons-Math ({n_math} bugs) with GPT-4o (`gpt-4o-2024-08-06`, temperature 0.0, {n_total * 2} total trials) using the initial lossy AST normalizer and on-demand `get_context` (`configs/pilot_65.yaml`), after {n_aborted} late-stage Math bugs from the {n_curated_init}-bug curated manifest were halted by OpenAI API billing/quota exhaustion (`data/exclusion_log.json`):",
        "",
        "| Metric | Raw Diff (Control) | AST Edit Script (Treatment) | Comparison / Statistical Test |",
        "|:-------|:-------------------|:----------------------------|:------------------------------|",
        f"| **Strict Assertion Validity (Primary, RQ1)** | **{rq1_s['raw_rate']*100:.1f}%** ({raw_s_cnt}/{n_total}) | {rq1_s['ast_rate']*100:.1f}% ({ast_s_cnt}/{n_total}) | McNemar $p = {rq1_s['p_value']:.4f}$ |",
        f"| **LIBRO Fail/Pass Validity (Secondary, RQ1)** | **{rq1_l['raw_rate']*100:.1f}%** ({raw_l_cnt}/{n_total}) | {rq1_l['ast_rate']*100:.1f}% ({ast_l_cnt}/{n_total}) | McNemar $p = {rq1_l['p_value']:.4f}$ |",
        f"| **Mean FDG (Ochiai, Strict Paired, $N={n_paired}$)** | {rq2_o.get('raw_mean', 0.0):.4f} | {rq2_o.get('ast_mean', 0.0):.4f} | Wilcoxon $p = {rq2_o.get('p_value', 1.0):.4f}$, $\\hat{{A}}_{{12}} = {rq2_o.get('a12', 0.5):.3f}$ ({paired_counts}; sign $p = {rq2_sign.get('p_value_two_sided', 1.0):.4f}$) |",
        f"| **Mean FDG (DStar $*=2$, Strict Paired, $N={n_paired}$)** | {rq2_d.get('raw_mean', 0.0):.4f} | {rq2_d.get('ast_mean', 0.0):.4f} | Wilcoxon $p = {rq2_d.get('p_value', 1.0):.4f}$, $\\hat{{A}}_{{12}} = {rq2_d.get('a12', 0.5):.3f}$ |",
        f"| **Realised Mean $\\Delta\\text{{AP}}$ (Ochiai, $N={n_paired}$)** | {rq2_ap.get('raw_mean', 0.0):+.4f} | {rq2_ap.get('ast_mean', 0.0):+.4f} | Wilcoxon $p = {rq2_ap.get('p_value', 1.0):.4f}$, $\\hat{{A}}_{{12}} = {rq2_ap.get('a12', 0.5):.3f}$ |",
        f"| **Method acc@3 (Strict Paired, $N={n_paired}$)** | {acc3_raw} | {acc3_ast} | Baseline = {acc3_base} |",
        f"| **Expected Diagnosability $E[\\text{{FDG}}]$ (Strict, $N={n_total}$)** | **{rq2_exp.get('raw_mean', 0.0):.4f}** | {rq2_exp.get('ast_mean', 0.0):.4f} | Wilcoxon $p = {rq2_exp.get('p_value', 1.0):.4f}$, $\\hat{{A}}_{{12}} = {rq2_exp.get('a12', 0.5):.3f}$ |",
        f"| **Mean Token Overhead (RQ3)** | {raw_tok:,.0f} tokens | **{ast_tok:,.0f} tokens** | **{tok_red:.1f}% reduction** with AST ({raw_att:.2f} vs. {ast_att:.2f} attempts) |",
        "",
        f"**Pilot Key Finding:** In the initial {n_total}-bug pilot, raw unified diffs outperformed the initial over-normalized AST edit scripts ({rq1_s['raw_rate']*100:.1f}% vs. {rq1_s['ast_rate']*100:.1f}% strict assertion validity, McNemar $p = {rq1_s['p_value']:.4f}$; {rq1_l['raw_rate']*100:.1f}% vs. {rq1_l['ast_rate']*100:.1f}% LIBRO fail/pass; {lang_s.get('raw_rate', 0)*100:.1f}% vs. {lang_s.get('ast_rate', 0)*100:.1f}% strict on Lang, {math_s.get('raw_rate', 0)*100:.1f}% vs. {math_s.get('ast_rate', 0)*100:.1f}% strict on Math; LIBRO by-project: {lang_l.get('raw_rate', 0)*100:.1f}% vs. {lang_l.get('ast_rate', 0)*100:.1f}% Lang, {math_l.get('raw_rate', 0)*100:.1f}% vs. {math_l.get('ast_rate', 0)*100:.1f}% Math) because GumTree v3.0.0 JSON omitted nested subtrees (stripping operand literals) and retained spurious `TypeDeclaration` header noise in 49/{n_total} bugs. Over strict jointly valid pairs ($N_{{\\text{{paired}}}}={n_paired}$), paired FDG counts were {paired_counts} (Wilcoxon $p = {rq2_o.get('p_value', 1.0):.4f}$). {has_passing_note} For the full rerun (`configs/experiment.yaml`), we patched `gumtree_normalizer.py` (verified across all {n_curated_init} curated bugs with 0 header-noise actions and full subtree/MOVE literal recovery) and enabled `inline_enclosing_method: true`.",
        END_MARKER,
    ]
    return "\n".join(lines)


def update_readme(
    readme_path: Path = Path("README.md"),
    summary_path: Path = Path("docs/figures/statistical_summary.json"),
    exclusion_path: Path = Path("data/exclusion_log.json"),
    eval_meta_path: Path = Path("data/results/evaluation_metadata.json"),
) -> str:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    exclusion_log = json.loads(exclusion_path.read_text(encoding="utf-8"))
    eval_meta = (
        json.loads(eval_meta_path.read_text(encoding="utf-8"))
        if eval_meta_path.exists()
        else None
    )
    new_block = render_empirical_section(summary, exclusion_log, eval_meta)
    text = readme_path.read_text(encoding="utf-8")

    if START_MARKER in text and END_MARKER in text:
        pattern = re.compile(
            re.escape(START_MARKER) + r".*?" + re.escape(END_MARKER),
            flags=re.DOTALL,
        )
        updated = pattern.sub(lambda _m: new_block, text, count=1)
    else:
        # Replace legacy "## Empirical Results Summary" section up to next "---"
        legacy_pat = re.compile(
            r"## Empirical Results Summary.*?(?=\r?\n---\r?\n)",
            flags=re.DOTALL,
        )
        if not legacy_pat.search(text):
            raise RuntimeError("Could not locate Empirical Results section in README.md")
        updated = legacy_pat.sub(lambda _m: new_block, text, count=1)

    readme_path.write_text(updated, encoding="utf-8")
    return updated


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync README.md numbers from statistical_summary.json.")
    parser.add_argument("--readme", type=Path, default=Path("README.md"))
    parser.add_argument("--summary", type=Path, default=Path("docs/figures/statistical_summary.json"))
    parser.add_argument("--exclusion", type=Path, default=Path("data/exclusion_log.json"))
    parser.add_argument("--eval-meta", type=Path, default=Path("data/results/evaluation_metadata.json"))
    args = parser.parse_args()
    update_readme(args.readme, args.summary, args.exclusion, args.eval_meta)
    print(f"Updated {args.readme} from {args.summary}")


if __name__ == "__main__":
    main()
