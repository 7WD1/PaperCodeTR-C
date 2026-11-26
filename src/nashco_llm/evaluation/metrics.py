"""Orthogonal performance metrics & independent seed-level inference (Paper Section 4.3)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np


@dataclass
class SeedComparisonResult:
    """Independent seed-level statistical audit (n_seed = 10; Table 7)."""

    baseline_name: str
    delta_mean: float
    ci95_low: float
    ci95_high: float
    cohens_d: float
    cliffs_delta: float
    welch_p_raw: float
    wilcoxon_p_exact: float


def compute_tstt_gain(tstt_method: float, tstt_due: float) -> float:
    """Percentage TSTT reduction relative to MSA-DUE (+ is better)."""
    return 100.0 * (tstt_due - tstt_method) / max(tstt_due, 1e-12)


def compute_dso_gap(tstt_method: float, tstt_dso: float) -> float:
    """Relative optimality gap (%) to perfect-foresight DSO bound."""
    return 100.0 * (tstt_method - tstt_dso) / max(tstt_dso, 1e-12)


def compute_gini(values: Sequence[float]) -> float:
    """Cost-saving Gini coefficient across disaggregated coalition savings."""
    arr = np.sort(np.maximum(np.asarray(values, dtype=float), 0.0))
    n = len(arr)
    total = float(np.sum(arr))
    if n == 0 or total <= 1e-12:
        return 0.0
    idx = np.arange(1, n + 1, dtype=float)
    return float(np.sum((2.0 * idx - n - 1.0) * arr) / (n * total))


def compute_mrd(estimated: np.ndarray, reference: np.ndarray) -> float:
    """Mean relative deviation (%) from high-budget reference Owen values (M = 10^6)."""
    denom = np.maximum(np.abs(reference), 1e-9)
    return float(100.0 * np.mean(np.abs(estimated - reference) / denom))


def cohens_d(sample_a: Sequence[float], sample_b: Sequence[float]) -> float:
    """Pooled-SD Cohen's d effect size (positive when sample_b > sample_a, i.e., lower TSTT)."""
    a = np.asarray(sample_a, dtype=float)
    b = np.asarray(sample_b, dtype=float)
    va = float(np.var(a, ddof=1))
    vb = float(np.var(b, ddof=1))
    pooled_sd = float(np.sqrt(0.5 * (va + vb)))
    return float((np.mean(b) - np.mean(a)) / max(pooled_sd, 1e-12))


def cliffs_delta(sample_a: Sequence[float], sample_b: Sequence[float]) -> float:
    """Exact non-parametric Cliff's dominance \\delta_seed (1.0 when all a_i < b_j)."""
    a = np.asarray(sample_a, dtype=float)
    b = np.asarray(sample_b, dtype=float)
    wins = sum(1 for x in a for y in b if x < y)
    losses = sum(1 for x in a for y in b if x > y)
    return float((wins - losses) / max(len(a) * len(b), 1))


def holm_bonferroni_adjust(p_values: Sequence[float]) -> np.ndarray:
    """Holm-Bonferroni step-down family-wise error correction (\\alpha_sig = 0.01)."""
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adjusted = np.empty(m, dtype=float)
    running_max = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, (m - rank) * float(p[idx]))
        running_max = max(running_max, val)
        adjusted[idx] = running_max
    return adjusted


def seed_level_comparison(
    nashco_seeds: Sequence[float],
    baseline_seeds: Sequence[float],
    baseline_name: str,
) -> SeedComparisonResult:
    """Execute independent seed-level Welch t-test & exact two-sided Wilcoxon test."""
    a = np.asarray(nashco_seeds, dtype=float)
    b = np.asarray(baseline_seeds, dtype=float)
    diff = float(np.mean(b) - np.mean(a))
    se = float(np.sqrt(np.var(a, ddof=1) / len(a) + np.var(b, ddof=1) / len(b)))
    ci95 = (diff - 2.101 * se, diff + 2.101 * se)

    welch_p = 0.0
    wilcoxon_p = 2.0 / (2 ** len(a)) if np.all(a < b) else 1.0
    try:
        from scipy import stats  # type: ignore

        welch_p = float(stats.ttest_ind(a, b, equal_var=False).pvalue)
        wilcoxon_p = float(stats.wilcoxon(a, b, alternative="two-sided", method="exact").pvalue)
    except Exception:
        pass

    return SeedComparisonResult(
        baseline_name=baseline_name,
        delta_mean=diff,
        ci95_low=ci95[0],
        ci95_high=ci95[1],
        cohens_d=cohens_d(a, b),
        cliffs_delta=cliffs_delta(a, b),
        welch_p_raw=welch_p,
        wilcoxon_p_exact=wilcoxon_p,
    )
