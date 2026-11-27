"""Unit tests for McNemar's test, Wilcoxon signed-rank, Vargha-Delaney A12, and plot/LaTeX output."""

from pathlib import Path

from src.evaluation.stats_and_plots import (
    analyze_evaluation_records,
    mcnemar_exact,
    paired_wilcoxon_summary,
    vargha_delaney_a12,
)


def test_mcnemar_and_a12():
    res = mcnemar_exact([True, False, False, False, False, False, False],
                        [True, True, True, True, True, True, True])
    assert res["ast_only_valid"] == 6
    assert res["raw_only_valid"] == 0
    assert res["p_value"] < 0.05

    assert vargha_delaney_a12([0.8, 0.9, 0.7], [0.2, 0.3, 0.4]) == 1.0
    assert vargha_delaney_a12([0.5, 0.5], [0.5, 0.5]) == 0.5


def test_analyze_evaluation_records_generates_tables_and_plots(tmp_path: Path):
    records = [
        {
            "bug_key": f"Lang-{i}",
            "project": "Lang",
            "verdicts": {
                "raw_diff": {"is_valid_libro": True, "is_valid_strict_assertion": True},
                "ast_edit": {"is_valid_libro": True, "is_valid_strict_assertion": True},
            },
            "fdg": {
                "raw_diff": {
                    "ochiai": {"fdg": 0.45 + 0.01 * i, "delta_ap": 0.05, "baseline_best_rank": 5, "augmented_best_rank": 3},
                    "dstar": {"fdg": 0.44 + 0.01 * i, "delta_ap": 0.04},
                },
                "ast_edit": {
                    "ochiai": {"fdg": 0.62 + 0.01 * i, "delta_ap": 0.14, "baseline_best_rank": 5, "augmented_best_rank": 1},
                    "dstar": {"fdg": 0.60 + 0.01 * i, "delta_ap": 0.12},
                },
            },
        }
        for i in range(1, 7)
    ]

    summary = analyze_evaluation_records(records, tmp_path)
    assert summary["rq2_fdg_ochiai_paired"]["n_paired"] == 6
    assert summary["rq2_fdg_ochiai_paired"]["a12"] > 0.71
    assert "rq2_expected_fdg_global" in summary
    assert summary["rq2_expected_fdg_global"]["n_paired"] == 6
    assert "rq1_by_project" in summary and "Lang" in summary["rq1_by_project"]
    assert "rq1_by_project_strict" in summary and "Lang" in summary["rq1_by_project_strict"]
    assert "rq1_failure_modes" in summary
    assert "rq2_sign_test_ochiai" in summary
    assert summary["rq2_sign_test_ochiai"]["n_positive_ast"] == 6
    assert "rq2_tost_ochiai" not in summary
    assert "rq2_acc_at_n" in summary
    assert summary["rq2_acc_at_n"]["acc@1"]["ast_edit"] == 6
    assert (tmp_path / "table_rq1.tex").exists()
    assert (tmp_path / "table_rq2.tex").exists()
    assert (tmp_path / "table_summary_metrics.tex").exists()
    assert (tmp_path / "rq1_validity_comparison.png").exists()
    assert (tmp_path / "rq2_fdg_paired.png").exists()


def test_analyze_evaluation_filters_unexecuted_bugs(tmp_path: Path):
    records = [
        {
            "bug_key": "Lang-1",
            "project": "Lang",
            "verdicts": {
                "raw_diff": {"is_valid_libro": True, "is_valid_strict_assertion": True},
                "ast_edit": {"is_valid_libro": True, "is_valid_strict_assertion": True},
            },
            "fdg": {},
        },
        {
            "bug_key": "Lang-2",
            "project": "Lang",
            "verdicts": {},  # Unexecuted bug
            "fdg": {},
        },
    ]
    summary = analyze_evaluation_records(records, tmp_path)
    # Total evaluated should be 1, not 2
    assert summary["rq1_libro_validity"]["n_total"] == 1
    assert summary["rq1_libro_validity"]["both_valid"] == 1


def test_update_readme_renders_summary_from_json(tmp_path: Path):
    import json
    from scripts.update_readme_stats import render_empirical_section, update_readme

    summary = {
        "rq1_strict_assertion_validity": {"n_total": 10, "both_valid": 2, "raw_only_valid": 3, "ast_only_valid": 1, "raw_rate": 0.5, "ast_rate": 0.3, "p_value": 0.625},
        "rq1_libro_validity": {"n_total": 10, "both_valid": 3, "raw_only_valid": 3, "ast_only_valid": 1, "raw_rate": 0.6, "ast_rate": 0.4, "p_value": 0.625},
        "rq1_by_project_strict": {"Lang": {"n_total": 10, "raw_rate": 0.5, "ast_rate": 0.3}},
        "rq1_by_project": {"Lang": {"n_total": 10, "raw_rate": 0.6, "ast_rate": 0.4}},
        "rq2_fdg_ochiai_paired": {"n_paired": 2, "raw_mean": 0.42, "ast_mean": 0.44, "p_value": 0.5, "a12": 0.6},
        "rq2_fdg_dstar_paired": {"n_paired": 2, "raw_mean": 0.41, "ast_mean": 0.43, "p_value": 0.5, "a12": 0.6},
        "rq2_delta_ap_ochiai_paired": {"n_paired": 2, "raw_mean": 0.05, "ast_mean": 0.08, "p_value": 0.5, "a12": 0.6},
        "rq2_sign_test_ochiai": {"n_positive_ast": 1, "n_exact_ties": 1, "n_negative_raw": 0, "p_value_two_sided": 1.0},
        "rq2_expected_fdg_global": {"raw_mean": 0.21, "ast_mean": 0.13, "p_value": 0.25, "a12": 0.45},
        "rq2_acc_at_n": {"acc@3": {"baseline": 1, "raw_diff": 2, "ast_edit": 2, "delta_raw": 1, "delta_ast": 1}},
        "rq3_efficiency": {"raw_diff": {"mean_tokens": 20000, "mean_attempts": 2.5}, "ast_edit": {"mean_tokens": 18000, "mean_attempts": 2.6}, "token_reduction_pct": 10.0},
    }
    excl = {"total_curated_initial": 12, "exclusion_counts": {"api_quota_exhausted_aborted": 2}}
    s_path = tmp_path / "summary.json"
    e_path = tmp_path / "excl.json"
    r_path = tmp_path / "README.md"
    s_path.write_text(json.dumps(summary), encoding="utf-8")
    e_path.write_text(json.dumps(excl), encoding="utf-8")
    r_path.write_text("# Title\n\n---\n\n## Empirical Results Summary\nold\n\n---\n\n## Footer\n", encoding="utf-8")

    out = update_readme(r_path, s_path, e_path, tmp_path / "missing_meta.json")
    assert "50.0%" in out and "30.0%" in out
    assert "1 AST wins / 1 ties / 0 raw wins" in out
    assert "TOST" not in out
