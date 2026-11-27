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
    assert "rq1_failure_modes" in summary
    assert "rq2_sign_test_ochiai" in summary
    assert "p1" in summary["rq2_tost_ochiai"] and "p2" in summary["rq2_tost_ochiai"]
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
