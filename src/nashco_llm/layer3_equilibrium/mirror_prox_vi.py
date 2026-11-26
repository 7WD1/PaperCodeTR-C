"""Entropy-regularized multi-player VI & two-step composite Mirror-Prox (Section 3.2.2).

Implements:
- Eq. (6): Normalized path shares p_{p,m}(t) = f_{p,m}(t) / d_{w,m}(t) on simplex \\Delta.
- Eqs. (7)-(8): 4-class VI operators (HDV SUE, CAV UE, Cournot-Nash FO, Cooperative with
  exact atomic self-externality \\partial C_i / \\partial f_{p,i} + \\sum_{a in p} \\pi_{a,i}).
- Eqs. (9)-(10): Implicit two-step composite Mirror-Prox projection (Theorems 1 & 2).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

import numpy as np

from ..config import EquilibriumConfig
from ..types import PerturbationTensor, UserClass
from .traffic_cost import SemanticTrafficCostModel


@dataclass
class ODClassBlock:
    """Single probability sub-simplex \\Delta_{w,m}(t) in Eq. (6)."""

    od_id: int
    user_class: UserClass
    class_idx: int
    demand: float
    path_indices: List[int]
    is_atomic_fleet: bool = False
    fleet_member_id: Optional[int] = None


@dataclass
class VIResult:
    """Converged equilibrium output and diagnostics."""

    path_shares: np.ndarray          # Shape (num_blocks, max_paths) ragged -> stored as list/flat
    path_flows_by_class: np.ndarray  # Shape (|P|, |N|)
    link_flows: np.ndarray           # Shape (|A|,)
    class_link_flows: np.ndarray     # Shape (|A|, |N|)
    iterations: int
    final_residual: float
    kl_trajectory: List[float]
    converged: bool


class CompositeMirrorProxSolver:
    """Solves Eq. (7) via two-step implicit composite Mirror-Prox Eqs. (9)-(10)."""

    def __init__(
        self,
        cost_model: SemanticTrafficCostModel,
        path_link_incidence: np.ndarray,
        blocks: Sequence[ODClassBlock],
        alpha_vot: np.ndarray,
        config: EquilibriumConfig,
    ) -> None:
        self.cost_model = cost_model
        self.delta_inc = np.asarray(path_link_incidence, dtype=float)  # (|P|, |A|)
        self.blocks = list(blocks)
        self.alpha_vot = np.asarray(alpha_vot, dtype=float)
        self.cfg = config
        self.num_paths, self.num_links = self.delta_inc.shape
        self.num_classes = len(self.alpha_vot)

        # Class-specific entropy weight \tau_m: HDV uses \theta_HDV^{-1}, others use \tau
        self.block_tau = np.array(
            [
                (1.0 / self.cfg.theta_hdv) if b.user_class == UserClass.HDV else self.cfg.entropy_tau
                for b in self.blocks
            ],
            dtype=float,
        )

    def uniform_initial_shares(self) -> List[np.ndarray]:
        """Initialize strictly interior uniform shares p_{p,m}^{(0)} = 1 / |P_w|."""
        return [np.full(len(b.path_indices), 1.0 / len(b.path_indices), dtype=float) for b in self.blocks]

    def shares_to_flows(self, shares: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Map normalized shares p \\in \\Delta to (path_flows_by_class, link_flows, class_link_flows)."""
        path_flows = np.zeros((self.num_paths, self.num_classes), dtype=float)
        for b, p_vec in zip(self.blocks, shares):
            for local_idx, pid in enumerate(b.path_indices):
                path_flows[pid, b.class_idx] += b.demand * float(p_vec[local_idx])
        class_link_flows = self.delta_inc.T @ path_flows  # (|A|, |N|)
        link_flows = np.sum(class_link_flows, axis=1)     # (|A|,)
        return path_flows, link_flows, class_link_flows

    def evaluate_operator(
        self,
        shares: Sequence[np.ndarray],
        xi: PerturbationTensor,
        msc_rates_fn: Optional[Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]] = None,
        dnl_override_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> tuple[List[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate Eq. (8) unregularized marginal operator \\Phi_{p,m}(p; \\Xi).

        Includes exact atomic self-externality \\sum_{a in p} \\alpha_q x_{a,q} dc_a/dx_a
        for Cournot-Nash FO fleets and atomic cooperative fleets, plus link MSC rates \\pi_{a,i}.
        """
        path_flows, link_flows, class_link_flows = self.shares_to_flows(shares)
        ext_tt = dnl_override_fn(link_flows) if dnl_override_fn is not None else None
        link_costs = self.cost_model.link_augmented_cost(link_flows, xi, external_travel_times=ext_tt)
        dc_dx = self.cost_model.link_cost_derivative(link_flows, xi)
        path_costs = self.cost_model.path_generalized_costs(link_costs, self.delta_inc, self.alpha_vot, xi)

        # Optional link-level MSC credit rates \pi_{a,m}, shape (|A|, |N|)
        if msc_rates_fn is not None:
            pi_matrix = msc_rates_fn(link_flows, class_link_flows, dc_dx)
        else:
            pi_matrix = np.zeros((self.num_links, self.num_classes), dtype=float)

        op_blocks: List[np.ndarray] = []
        for b in self.blocks:
            pids = b.path_indices
            m = b.class_idx
            phi_vec = path_costs[pids, m].copy()

            # Atomic fleet self-externality: \sum_{a in p} \alpha_q x_{a,q} (\partial \tilde{c}_a / \partial x_a)
            if b.user_class == UserClass.FO or (b.user_class == UserClass.COOP and b.is_atomic_fleet):
                self_ext_link = self.alpha_vot[m] * class_link_flows[:, m] * dc_dx
                phi_vec += self.delta_inc[pids] @ self_ext_link

            # Cooperative Stage-1 link MSC mobility-credit rate \sum_{a in p} \pi_{a,i}(t)
            if b.user_class == UserClass.COOP:
                phi_vec += self.delta_inc[pids] @ pi_matrix[:, m]

            op_blocks.append(phi_vec)

        return op_blocks, path_flows, link_flows, class_link_flows

    @staticmethod
    def _implicit_entropic_step(
        p_curr: np.ndarray,
        phi_vec: np.ndarray,
        gamma_s: float,
        tau_m: float,
    ) -> np.ndarray:
        """Closed-form implicit composite proximal step (Eqs. 9 & 10) in log-domain for stability."""
        beta = 1.0 / (1.0 + gamma_s * tau_m)
        log_p = np.log(np.maximum(p_curr, 1e-30))
        logits = beta * (log_p - gamma_s * phi_vec)
        logits -= np.max(logits)
        weights = np.exp(logits)
        return weights / np.sum(weights)

    def demand_weighted_kl(self, p_a: Sequence[np.ndarray], p_b: Sequence[np.ndarray]) -> float:
        """Demand-weighted Bregman KL Lyapunov function D_{h_D}(p_a, p_b) (Theorem 2)."""
        total = 0.0
        for b, pa, pb in zip(self.blocks, p_a, p_b):
            pa_safe = np.maximum(pa, 1e-30)
            pb_safe = np.maximum(pb, 1e-30)
            total += b.demand * float(np.sum(pa_safe * np.log(pa_safe / pb_safe)))
        return total

    def solve(
        self,
        xi: PerturbationTensor,
        initial_shares: Optional[Sequence[np.ndarray]] = None,
        msc_rates_fn: Optional[Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]] = None,
        dnl_override_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        tolerance: Optional[float] = None,
        constant_step: Optional[float] = None,
    ) -> VIResult:
        """Execute two-step composite Mirror-Prox loop until relative share residual <= \\epsilon_VI."""
        tol = self.cfg.vi_tolerance if tolerance is None else tolerance
        shares = (
            [np.asarray(p, dtype=float).copy() for p in initial_shares]
            if initial_shares is not None
            else self.uniform_initial_shares()
        )
        kl_hist: List[float] = []
        residual = float("inf")
        converged = False

        for s in range(self.cfg.max_vi_iters):
            gamma_s = (
                constant_step
                if constant_step is not None
                else self.cfg.mirror_prox_gamma0 / ((1.0 + s) ** self.cfg.mirror_prox_kappa)
            )
            # Step 1 (Eq. 9): Extrapolation half-step \tilde{p}^{(s)}
            phi_s, _, _, _ = self.evaluate_operator(shares, xi, msc_rates_fn, dnl_override_fn)
            p_tilde = [
                self._implicit_entropic_step(p_b, phi_b, gamma_s, tau_b)
                for p_b, phi_b, tau_b in zip(shares, phi_s, self.block_tau)
            ]

            # Step 2 (Eq. 10): Correction full-step p^{(s+1)}
            phi_tilde, _, _, _ = self.evaluate_operator(p_tilde, xi, msc_rates_fn, dnl_override_fn)
            p_next = [
                self._implicit_entropic_step(p_b, phi_tb, gamma_s, tau_b)
                for p_b, phi_tb, tau_b in zip(shares, phi_tilde, self.block_tau)
            ]

            # Block-weighted l2/l1 norm ||p_next - p_curr||_{D,2,1} (Theorem 2)
            num = 0.0
            den = 0.0
            for b, p_new, p_old in zip(self.blocks, p_next, shares):
                num += b.demand * (float(np.sum(np.abs(p_new - p_old))) ** 2)
                den += b.demand * (float(np.sum(np.abs(p_old))) ** 2)
            residual = float(np.sqrt(num / max(den, 1e-18)))
            kl_hist.append(self.demand_weighted_kl(p_next, shares))
            shares = p_next

            if residual <= tol:
                converged = True
                break

        path_flows, link_flows, class_link_flows = self.shares_to_flows(shares)
        return VIResult(
            path_shares=np.array(shares, dtype=object),
            path_flows_by_class=path_flows,
            link_flows=link_flows,
            class_link_flows=class_link_flows,
            iterations=len(kl_hist),
            final_residual=residual,
            kl_trajectory=kl_hist,
            converged=converged,
        )

    @staticmethod
    def compute_herding_index(
        link_flow_history: np.ndarray,
        no_vi_variance_anchor: float,
    ) -> float:
        """Compute cross-period oscillation index hi = Var_t(x^{(t)}) / Var_t(x^{no-VI})."""
        var_t = float(np.mean(np.var(link_flow_history, axis=0)))
        return var_t / max(no_vi_variance_anchor, 1e-12)
