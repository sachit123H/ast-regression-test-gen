"""Mathematical verification of SBFL (Ochiai, DStar *=2) and FDG against An & Yoo (ISSTA 2022) Figure 1."""

import math
import numpy as np

from src.evaluation.fdg import ambiguity_groups, compute_fdg
from src.evaluation.sbfl import (
    average_precision,
    compute_acc_at_n,
    compute_map,
    compute_suspiciousness,
    rank_with_max_tiebreaker,
)


def test_an_yoo_figure_1_ochiai_and_max_tiebreaker():
    """Verify exact Ochiai scores & max tie-breaker ranks from An & Yoo (ISSTA 2022) Figure 1."""
    # Elements: e1, e2, e3 (faulty), e4, e5
    # t1 (PASS=1): covers e5
    # t2 (FAIL=0): covers e1, e2, e3, e4
    # t3 (FAIL=0): covers e3, e4
    C_T = np.array(
        [
            [0, 0, 0, 0, 1],
            [1, 1, 1, 1, 0],
            [0, 0, 1, 1, 0],
        ],
        dtype=np.int64,
    )
    R_T = [1, 0, 0]  # 1=PASS, 0=FAIL

    scores = compute_suspiciousness(C_T, R_T, formula="ochiai")
    # Expected from Fig. 1: [0.71, 0.71, 1.00, 1.00, 0.00]
    assert math.isclose(scores[0], 1 / math.sqrt(2), abs_tol=1e-6)
    assert math.isclose(scores[1], 1 / math.sqrt(2), abs_tol=1e-6)
    assert math.isclose(scores[2], 1.0, abs_tol=1e-6)
    assert math.isclose(scores[3], 1.0, abs_tol=1e-6)
    assert math.isclose(scores[4], 0.0, abs_tol=1e-6)

    ranks = rank_with_max_tiebreaker({f"e{i+1}": float(scores[i]) for i in range(5)})
    # Expected from Fig. 1: e3=2, e4=2, e1=4, e2=4, e5=5
    assert ranks == {"e3": 2, "e4": 2, "e1": 4, "e2": 4, "e5": 5}

    # Ambiguity groups of T: {{e1, e2}, {e3, e4}, {e5}} -> 3 groups
    assert len(ambiguity_groups(C_T)) == 3


def test_fdg_prioritizes_t4_over_t2_and_t1_on_figure_1():
    """Verify FDG ranks t'_4 (splits {e3,e4}) higher than t'_2 (splits {e1,e2}) and t'_1 (no split)."""
    C_T = np.array(
        [
            [0, 0, 0, 0, 1],
            [1, 1, 1, 1, 0],
            [0, 0, 1, 1, 0],
        ],
        dtype=np.int64,
    )
    R_T = [1, 0, 0]

    t1_prime = [0, 0, 0, 0, 1]  # Covers only e5 (susp=0), breaks no ambiguity group
    t2_prime = [1, 0, 0, 0, 1]  # Breaks {e1, e2} (susp=0.71)
    t4_prime = [0, 0, 1, 0, 0]  # Breaks {e3, e4} (susp=1.00)

    fdg_t1 = compute_fdg(C_T, R_T, t1_prime, alpha=0.5, formula="ochiai")
    fdg_t2 = compute_fdg(C_T, R_T, t2_prime, alpha=0.5, formula="ochiai")
    fdg_t4 = compute_fdg(C_T, R_T, t4_prime, alpha=0.5, formula="ochiai")

    assert fdg_t4.fdg > fdg_t2.fdg > fdg_t1.fdg
    assert fdg_t4.split > fdg_t2.split > fdg_t1.split
    assert fdg_t4.num_ambiguity_groups_after == 4

    # Verify DStar (*=2) also computes cleanly from the exact same spectra
    fdg_dstar = compute_fdg(C_T, R_T, t4_prime, alpha=0.5, formula="dstar", dstar_star=2)
    assert 0.0 <= fdg_dstar.fdg <= 1.0
    assert fdg_dstar.split > fdg_t1.split


def test_average_precision_and_acc_at_n():
    # Example from An & Yoo Sec. 3.2.1: faulty methods at ranks 2 and 5 -> AP = 0.5*(1/2 + 2/5) = 0.45
    ranks = {"A": 2, "B": 5, "C": 1, "D": 3, "E": 4}
    ap = average_precision(ranks, {"A", "B"})
    assert math.isclose(ap, 0.45, abs_tol=1e-6)

    acc = compute_acc_at_n([1, 2, 5, 12], n_values=[1, 3, 5, 10])
    assert acc == {1: 1, 3: 2, 5: 3, 10: 3}
    assert math.isclose(compute_map([0.45, 1.0]), 0.725, abs_tol=1e-6)
