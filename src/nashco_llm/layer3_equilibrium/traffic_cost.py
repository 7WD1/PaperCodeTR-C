"""Semantic-augmented dynamic link and path cost model (Paper Section 3.2.2, Eqs. 4-5)."""

from __future__ import annotations

import numpy as np

from ..config import EquilibriumConfig
from ..types import PerturbationTensor


class SemanticTrafficCostModel:
    """Computes Eq. (4) semantic-augmented link impedance and analytical derivatives."""

    def __init__(
        self,
        free_flow_times: np.ndarray,
        nominal_capacities: np.ndarray,
        config: EquilibriumConfig,
    ) -> None:
        self.t0 = np.asarray(free_flow_times, dtype=float)
        self.cap = np.asarray(nominal_capacities, dtype=float)
        self.cfg = config
        self.beta_safe = self.cfg.beta_safe_ratio * self.t0

    def link_travel_time(self, flows: np.ndarray, phi: np.ndarray) -> np.ndarray:
        """Smooth BPR surrogate \\ell_a(x_a, \\phi_a C_a) = t_0 [1 + b (x_a / (\\phi_a C_a))^p]."""
        eff_cap = np.maximum(phi * self.cap, 1e-6)
        ratio = np.maximum(flows, 0.0) / eff_cap
        return self.t0 * (1.0 + self.cfg.bpr_b * np.power(ratio, self.cfg.bpr_p))

    def link_augmented_cost(
        self,
        flows: np.ndarray,
        xi: PerturbationTensor,
        external_travel_times: np.ndarray | None = None,
    ) -> np.ndarray:
        """Evaluate Eq. (4): \\tilde{c}_a = \\ell_a + \\beta_safe \\tilde{r}_a g_risk(x_a)."""
        eff_cap = np.maximum(xi.phi * self.cap, 1e-6)
        ratio = np.maximum(flows, 0.0) / eff_cap
        ell = (
            external_travel_times
            if external_travel_times is not None
            else self.link_travel_time(flows, xi.phi)
        )
        g_risk = 1.0 + self.cfg.risk_density_coeff * ratio
        return ell + self.beta_safe * xi.hazard_risk * g_risk

    def link_cost_derivative(
        self,
        flows: np.ndarray,
        xi: PerturbationTensor,
    ) -> np.ndarray:
        """Exact analytical derivative \\partial \\tilde{c}_a / \\partial x_a for MSC & surrogate."""
        eff_cap = np.maximum(xi.phi * self.cap, 1e-6)
        ratio = np.maximum(flows, 0.0) / eff_cap
        b, p = self.cfg.bpr_b, self.cfg.bpr_p
        d_ell = self.t0 * b * p * np.power(ratio, max(p - 1.0, 0.0)) / eff_cap
        d_risk = self.beta_safe * xi.hazard_risk * (self.cfg.risk_density_coeff / eff_cap)
        return d_ell + d_risk

    def path_generalized_costs(
        self,
        link_costs: np.ndarray,
        path_link_incidence: np.ndarray,
        alpha_vot: np.ndarray,
        xi: PerturbationTensor,
    ) -> np.ndarray:
        """Evaluate Eq. (5): C_{p,m} = \\alpha_m \\sum_{a in p} \\tilde{c}_a + \\zeta_{p,m}.

        Args:
            link_costs: Shape (|A|,)
            path_link_incidence: Binary matrix \\Delta_inc of shape (|P|, |A|)
            alpha_vot: Per-class value of time, shape (|N|,)
            xi: Verified perturbation tensor with zeta_bias of shape (|A|, |N|)

        Returns:
            Generalized path costs of shape (|P|, |N|).
        """
        base_path_cost = path_link_incidence @ link_costs  # (|P|,)
        zeta_path = path_link_incidence @ xi.zeta_bias     # (|P|, |N|)
        return np.outer(base_path_cost, alpha_vot) + zeta_path
