"""Stage 1: Revenue-recycling MSC mobility-credit pricing & adaptive law (Section 3.3.1).

Implements:
- Eq. (12): Unified network generalized social cost J_SO(f; \\Xi).
- Eq. (13): Non-atomic uniform rate \\eta_eff \\bar{\\alpha}_a x_a dc_a/dx_a and
  external-only atomic commercial-fleet rate \\eta_eff (\\bar{\\alpha}_a x_a - \\alpha_i x_{a,i}) dc_a/dx_a
  (unconditionally exact for heterogeneous values of time per Theorem 3(i)).
- Eq. (14): Adaptive compliance internalization law \\eta_eff(t) = clip(\\eta_0 \\rho^{-\\beta_\\eta}, \\eta_min, \\eta_max).
- Eq. (15) & retention update: Cooperative utility and bounded compliance feedback.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ..config import CooperativeConfig


class MSCCreditPricer:
    """Computes link-level MSC credit rates, J_SO, and adaptive compliance schedules."""

    def __init__(self, config: CooperativeConfig) -> None:
        self.cfg = config

    def compute_adaptive_eta(self, observed_compliance: float, use_full_tier1: bool = False) -> float:
        """Evaluate Eq. (14): adaptive internalization rate \\eta_eff(t)."""
        if use_full_tier1:
            return self.cfg.eta_eff_full
        if not self.cfg.adaptive_eta_enabled:
            return self.cfg.eta_eff_base
        rho = float(np.clip(observed_compliance, 0.05, 1.0))
        raw = self.cfg.eta_eff_base * (rho ** (-self.cfg.adaptive_beta))
        eta_min, eta_max = self.cfg.eta_bounds
        return float(np.clip(raw, eta_min, eta_max))

    def compute_link_msc_rates(
        self,
        link_flows: np.ndarray,
        class_link_flows: np.ndarray,
        dc_dx: np.ndarray,
        alpha_vot: np.ndarray,
        atomic_class_flags: Sequence[bool],
        eta_eff: float,
    ) -> np.ndarray:
        """Evaluate Eq. (13) link credit rate matrix \\pi_{a,m}(t) of shape (|A|, |N|).

        For non-atomic OD clusters:
            \\pi_{a,i} = \\eta_eff * (\\sum_m \\alpha_m x_{a,m}) * dc_a/dx_a.
        For atomic commercial fleets q:
            \\pi_{a,q}^{ext} = \\eta_eff * (\\sum_{m != q} \\alpha_m x_{a,m}) * dc_a/dx_a,
        preventing double-counting of fleet q's internalized self-externality (Theorem 3i).
        """
        num_links, num_classes = class_link_flows.shape
        # Flow-weighted total VOT mass \bar{\alpha}_a x_a = \sum_m \alpha_m x_{a,m}
        weighted_flow_sum = class_link_flows @ alpha_vot  # (|A|,)
        pi_matrix = np.zeros((num_links, num_classes), dtype=float)

        for m in range(num_classes):
            if atomic_class_flags[m]:
                ext_mass = np.maximum(weighted_flow_sum - alpha_vot[m] * class_link_flows[:, m], 0.0)
                pi_matrix[:, m] = eta_eff * ext_mass * dc_dx
            else:
                pi_matrix[:, m] = eta_eff * weighted_flow_sum * dc_dx
        return pi_matrix

    @staticmethod
    def compute_social_cost(
        class_link_flows: np.ndarray,
        link_costs: np.ndarray,
        alpha_vot: np.ndarray,
        path_flows_by_class: np.ndarray | None = None,
        zeta_path: np.ndarray | None = None,
    ) -> float:
        """Evaluate Eq. (12): J_SO(f; \\Xi) = \\sum_a \\bar{\\alpha}_a x_a \\tilde{c}_a + \\sum f \\zeta."""
        weighted_flows = class_link_flows @ alpha_vot
        j_so = float(np.sum(weighted_flows * link_costs))
        if path_flows_by_class is not None and zeta_path is not None:
            j_so += float(np.sum(path_flows_by_class * zeta_path))
        return j_so

    def update_compliance_retention(
        self,
        base_compliance: float,
        prev_core_violation: float,
    ) -> float:
        """Bounded inter-period compliance retention feedback \\rho(t) = clip(\\rho_0 - \\lambda \\varepsilon)."""
        updated = base_compliance - self.cfg.compliance_defection_gain * prev_core_violation
        return float(np.clip(updated, self.cfg.compliance_min, 1.0))
