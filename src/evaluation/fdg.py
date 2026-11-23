"""Exact Fault Diagnosability Gain (FDG) implementation (An & Yoo, ISSTA 2022).

Implements Equations from Section 4.1 of:
  Gabin An and Shin Yoo, "FDG: A Precise Measurement of Fault Diagnosability
  Gain of Test Cases", ISSTA 2022 (https://coinse.github.io/publications/pdfs/An2022pb.pdf)

Formulas:
  - `w_j`: min-max scaled suspiciousness score of element `e_j` under baseline `T`.
  - `p(e_j) = w_j / sum_i w_i` (uniform `1/n` if `sum_i w_i == 0`).
  - `p(g) = sum_{e_j in g} p(e_j)` for each ambiguity group `g in AG(T U {t})`.
  - `Split(T, t) = 1 - (1 / (n - 1)) * sum_{g in AG(T U {t})} p(g) * (|g| - 1)`
  - `Cover(T, t) = (1 / n) * sum_{j=1}^n w_j * c_t[j]`
  - `FDG(T, t)   = alpha * Split(T, t) + (1 - alpha) * Cover(T, t)` (default `alpha = 0.5`).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Dict, List, Sequence, Tuple

import numpy as np

from src.evaluation.sbfl import FormulaName, compute_suspiciousness, min_max_scale


@dataclass
class FDGResult:
    formula: FormulaName
    alpha: float
    split: float
    cover: float
    fdg: float
    num_elements: int
    num_ambiguity_groups_before: int
    num_ambiguity_groups_after: int

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def ambiguity_groups(coverage_matrix: np.ndarray) -> List[List[int]]:
    """Partition element indices `[0..n-1]` into ambiguity groups (identical coverage columns)."""
    C = np.asarray(coverage_matrix, dtype=np.int64)
    if C.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got shape {C.shape}")
    groups: Dict[Tuple[int, ...], List[int]] = defaultdict(list)
    n_elems = C.shape[1]
    for j in range(n_elems):
        col_sig = tuple(int(v) for v in C[:, j])
        groups[col_sig].append(j)
    return list(groups.values())


def compute_fdg(
    baseline_matrix: np.ndarray,
    baseline_results: Sequence[int],
    new_test_coverage: Sequence[int],
    alpha: float = 0.5,
    formula: FormulaName = "ochiai",
    dstar_star: int = 2,
) -> FDGResult:
    """Compute `Split(T, t)`, `Cover(T, t)`, and `FDG(T, t)` for adding test `t` to suite `T`.

    Args:
        baseline_matrix: Binary `(m, n)` coverage matrix `C_T` for existing suite `T`.
        baseline_results: Length-`m` results (`0` = FAIL, `1` = PASS) for `T`.
        new_test_coverage: Length-`n` binary coverage vector `c_t` for candidate test `t`.
        alpha: Weight on `Split` vs `Cover` (default `0.5` per An & Yoo ISSTA 2022).
        formula: Underlying SBFL formula (`"ochiai"` primary, `"dstar"` robustness).
        dstar_star: Exponent `*` for DStar (default `2`).
    """
    C_T = np.asarray(baseline_matrix, dtype=np.int64)
    c_t = np.asarray(new_test_coverage, dtype=np.int64).reshape(1, -1)
    if C_T.shape[1] != c_t.shape[1]:
        raise ValueError(
            f"Element count mismatch: baseline has {C_T.shape[1]}, new test has {c_t.shape[1]}"
        )
    n = C_T.shape[1]
    if n == 0:
        return FDGResult(
            formula=formula,
            alpha=alpha,
            split=0.0,
            cover=0.0,
            fdg=0.0,
            num_elements=0,
            num_ambiguity_groups_before=0,
            num_ambiguity_groups_after=0,
        )

    # 1. Min-max scaled suspiciousness scores `w_j` computed on baseline `T`
    raw_susp = compute_suspiciousness(
        C_T, baseline_results, formula=formula, dstar_star=dstar_star
    )
    w = min_max_scale(raw_susp)

    # 2. Fault probabilities `p(e_j) = w_j / sum_i w_i` (An & Yoo Sec. 4.1.1)
    w_sum = float(np.sum(w))
    if w_sum > 0.0:
        p = w / w_sum
    else:
        p = np.full(n, 1.0 / n, dtype=np.float64)

    # 3. Ambiguity groups before & after adding `t`
    ag_before = ambiguity_groups(C_T)
    C_aug = np.vstack([C_T, c_t])
    ag_after = ambiguity_groups(C_aug)

    # 4. Split(T, t) = 1 - (1 / (n - 1)) * sum_{g in AG(T U {t})} p(g) * (|g| - 1)
    if n <= 1:
        split_val = 1.0
    else:
        expected_waste = 0.0
        for g in ag_after:
            p_g = float(np.sum(p[g]))
            expected_waste += p_g * (len(g) - 1)
        split_val = 1.0 - (expected_waste / (n - 1))
        split_val = max(0.0, min(1.0, split_val))

    # 5. Cover(T, t) = (1 / n) * sum_{j=1}^n w_j * c_t[j]
    cover_val = float(np.dot(w, c_t.ravel())) / n

    # 6. FDG(T, t) = alpha * Split(T, t) + (1 - alpha) * Cover(T, t)
    fdg_val = alpha * split_val + (1.0 - alpha) * cover_val

    return FDGResult(
        formula=formula,
        alpha=alpha,
        split=round(split_val, 6),
        cover=round(cover_val, 6),
        fdg=round(fdg_val, 6),
        num_elements=n,
        num_ambiguity_groups_before=len(ag_before),
        num_ambiguity_groups_after=len(ag_after),
    )
