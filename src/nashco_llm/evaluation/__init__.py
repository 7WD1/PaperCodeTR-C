"""Evaluation metrics, seed-level statistical tests, and baseline registry (Section 4.3)."""

from .baselines import BASELINE_REGISTRY, BaselineSpec
from .metrics import (
    cliffs_delta,
    cohens_d,
    compute_gini,
    compute_mrd,
    compute_tstt_gain,
    holm_bonferroni_adjust,
    seed_level_comparison,
)

__all__ = [
    "BASELINE_REGISTRY",
    "BaselineSpec",
    "compute_tstt_gain",
    "compute_gini",
    "compute_mrd",
    "cohens_d",
    "cliffs_delta",
    "holm_bonferroni_adjust",
    "seed_level_comparison",
]
