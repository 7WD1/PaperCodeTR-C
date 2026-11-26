"""Guardrailed dual-agent behavioral calibrator (Paper Section 3.2.2, Eq. 11).

Executes every 3 rolling periods to align class VOT weights \\alpha_m and semantic path
biases \\zeta_{p,m} with observed compliance \\rho(t) while staying inside the compact
parameter box required by Theorem 1.
"""

from __future__ import annotations

import numpy as np

from ..config import SemanticConfig


class DualAgentCalibrator:
    """Bounded projected update (Eq. 11) with damping gain \\lambda_cal = 0.25."""

    def __init__(self, config: SemanticConfig) -> None:
        self.cfg = config

    def should_calibrate(self, period: int) -> bool:
        return period > 0 and (period % self.cfg.calib_interval_steps == 0)

    def update(
        self,
        alpha_vot: np.ndarray,
        zeta_bias: np.ndarray,
        observed_compliance: float,
        target_compliance: float = 0.95,
        delta_alpha_llm: np.ndarray | None = None,
        delta_zeta_llm: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply Eq. (11): clip(\\alpha + \\lambda \\Delta\\alpha, \\alpha_min, \\alpha_max)."""
        gap = target_compliance - observed_compliance
        d_alpha = (
            np.asarray(delta_alpha_llm, dtype=float)
            if delta_alpha_llm is not None
            else np.full_like(alpha_vot, 0.2 * gap, dtype=float)
        )
        d_zeta = (
            np.asarray(delta_zeta_llm, dtype=float)
            if delta_zeta_llm is not None
            else np.full_like(zeta_bias, -0.05 * gap, dtype=float)
        )

        alpha_min, alpha_max = self.cfg.alpha_bounds
        alpha_next = np.clip(alpha_vot + self.cfg.calib_lambda * d_alpha, alpha_min, alpha_max)
        zeta_next = np.clip(
            zeta_bias + self.cfg.calib_lambda * d_zeta,
            -self.cfg.zeta_max,
            self.cfg.zeta_max,
        )
        return alpha_next, zeta_next
