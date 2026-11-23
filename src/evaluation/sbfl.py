"""Spectrum-Based Fault Localisation (SBFL): Ochiai, DStar (*=2), method aggregation, acc@n, mAP.

Follows An & Yoo (ISSTA 2022) Section 2.2 & Section 3.2.1 and Proposal Section 4.6:
- Line-level suspiciousness via Ochiai (primary) and DStar (*=2, robustness).
- Min-max scaling of suspiciousness scores to [0, 1] (An & Yoo Footnote 2).
- Method-level aggregation by taking the maximum suspiciousness score among all
  lines belonging to the method.
- Ranking with the max tie-breaker (all tied methods share the worst/highest
  rank index in their tie group, matching Figure 1 of An & Yoo ISSTA 2022).
- Evaluation via acc@n (n in {1, 3, 5, 10}) and Average Precision / mAP.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Literal, Sequence, Set, Tuple

import numpy as np


FormulaName = Literal["ochiai", "dstar"]


@dataclass
class SpectrumCounts:
    ep: np.ndarray  # Passing tests covering element j
    np_: np.ndarray # Passing tests NOT covering element j
    ef: np.ndarray  # Failing tests covering element j
    nf: np.ndarray  # Failing tests NOT covering element j


def compute_spectrum(
    coverage_matrix: np.ndarray,
    test_results: Sequence[int],
) -> SpectrumCounts:
    """Compute (ep, np, ef, nf) per program element from an `m x n` binary matrix.

    Args:
        coverage_matrix: Binary array of shape `(m, n)` where `1` means covered.
        test_results: Length-`m` sequence where `0` = FAIL and `1` = PASS
                      (matching An & Yoo ISSTA 2022 Section 2.1: R(t)=0 if failing).
    """
    C = np.asarray(coverage_matrix, dtype=np.int64)
    if C.ndim != 2:
        raise ValueError(f"Expected 2D coverage matrix, got shape {C.shape}")
    R = np.asarray(test_results, dtype=np.int64)
    if R.shape[0] != C.shape[0]:
        raise ValueError(f"Row count mismatch: C has {C.shape[0]} rows, R has {R.shape[0]}")

    fail_mask = R == 0
    pass_mask = R == 1

    ef = C[fail_mask].sum(axis=0) if np.any(fail_mask) else np.zeros(C.shape[1], dtype=np.int64)
    nf = (1 - C[fail_mask]).sum(axis=0) if np.any(fail_mask) else np.zeros(C.shape[1], dtype=np.int64)
    ep = C[pass_mask].sum(axis=0) if np.any(pass_mask) else np.zeros(C.shape[1], dtype=np.int64)
    np_ = (1 - C[pass_mask]).sum(axis=0) if np.any(pass_mask) else np.zeros(C.shape[1], dtype=np.int64)
    return SpectrumCounts(ep=ep, np_=np_, ef=ef, nf=nf)


def ochiai_scores(spec: SpectrumCounts) -> np.ndarray:
    """Ochiai(ep, np, ef, nf) = ef / sqrt((ef + nf) * (ef + ep))."""
    denom = np.sqrt((spec.ef + spec.nf).astype(np.float64) * (spec.ef + spec.ep).astype(np.float64))
    out = np.zeros_like(denom, dtype=np.float64)
    valid = denom > 0
    out[valid] = spec.ef[valid] / denom[valid]
    return out


def dstar_scores(spec: SpectrumCounts, star: int = 2) -> np.ndarray:
    """DStar(ep, np, ef, nf) = (ef ** star) / (ep + nf) (Wong et al., 2014).

    When `ep + nf == 0` and `ef > 0`, assigns a finite sentinel strictly above the
    maximum finite score so min-max scaling maps those elements cleanly to 1.0.
    """
    num = np.power(spec.ef.astype(np.float64), star)
    denom = (spec.ep + spec.nf).astype(np.float64)
    out = np.zeros_like(num, dtype=np.float64)

    finite_mask = denom > 0
    out[finite_mask] = num[finite_mask] / denom[finite_mask]

    inf_mask = (denom == 0) & (spec.ef > 0)
    if np.any(inf_mask):
        max_finite = float(np.max(out[finite_mask])) if np.any(finite_mask) else 0.0
        out[inf_mask] = max_finite + 1.0
    return out


def compute_suspiciousness(
    coverage_matrix: np.ndarray,
    test_results: Sequence[int],
    formula: FormulaName = "ochiai",
    dstar_star: int = 2,
) -> np.ndarray:
    """Compute raw suspiciousness scores for every program element."""
    spec = compute_spectrum(coverage_matrix, test_results)
    if formula == "ochiai":
        return ochiai_scores(spec)
    if formula == "dstar":
        return dstar_scores(spec, star=dstar_star)
    raise ValueError(f"Unsupported SBFL formula: {formula!r}")


def min_max_scale(scores: np.ndarray) -> np.ndarray:
    """Min-max scale scores to `[0, 1]` (An & Yoo ISSTA 2022 Footnote 2)."""
    arr = np.asarray(scores, dtype=np.float64)
    if arr.size == 0:
        return arr
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if math.isclose(hi, lo, rel_tol=1e-12, abs_tol=1e-12):
        # If all scores are identical and positive, map to 1.0; if all zero, map to 0.0
        return np.ones_like(arr) if hi > 0.0 else np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def aggregate_line_to_method_scores(
    elements: Sequence[str],
    line_scores: np.ndarray,
    line_to_method: Dict[str, str],
) -> Dict[str, float]:
    """Aggregate line-level suspiciousness to method level via maximum line score."""
    method_scores: Dict[str, float] = {}
    for elem, score in zip(elements, line_scores):
        m_key = line_to_method.get(elem, elem)
        val = float(score)
        if m_key not in method_scores or val > method_scores[m_key]:
            method_scores[m_key] = val
    return method_scores


def rank_with_max_tiebreaker(scores: Dict[str, float], tol: float = 1e-12) -> Dict[str, int]:
    """Assign 1-indexed ranks using the max tie-breaker (worst rank in tie group).

    Example (An & Yoo Fig. 1): scores `{e3: 1.0, e4: 1.0, e1: 0.71, e2: 0.71, e5: 0.0}`
    -> ranks `{e3: 2, e4: 2, e1: 4, e2: 4, e5: 5}`.
    """
    items = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    ranks: Dict[str, int] = {}
    i = 0
    n = len(items)
    while i < n:
        j = i + 1
        while j < n and math.isclose(items[j][1], items[i][1], rel_tol=tol, abs_tol=tol):
            j += 1
        # All items in [i, j) share the max rank `j` (1-indexed)
        for k in range(i, j):
            ranks[items[k][0]] = j
        i = j
    return ranks


def average_precision(ranks: Dict[str, int], faulty_elements: Set[str]) -> float:
    """Compute Average Precision (AP) over faulty methods for a single bug."""
    present_ranks = sorted(ranks[f] for f in faulty_elements if f in ranks)
    if not present_ranks:
        return 0.0
    precisions = [(idx + 1) / r for idx, r in enumerate(present_ranks)]
    return float(sum(precisions) / len(faulty_elements))


def best_faulty_rank(ranks: Dict[str, int], faulty_elements: Set[str]) -> int:
    """Return the best (lowest) rank achieved by any faulty element, or `len(ranks) + 1`."""
    present = [ranks[f] for f in faulty_elements if f in ranks]
    return min(present) if present else (len(ranks) + 1)


def compute_acc_at_n(
    best_ranks: Iterable[int],
    n_values: Sequence[int] = (1, 3, 5, 10),
) -> Dict[int, int]:
    """Count how many bugs have at least one faulty method ranked `<= n`."""
    ranks_list = list(best_ranks)
    return {n: sum(1 for r in ranks_list if r <= n) for n in n_values}


def compute_map(aps: Iterable[float]) -> float:
    """Mean Average Precision across a set of bugs."""
    vals = list(aps)
    return float(sum(vals) / len(vals)) if vals else 0.0
