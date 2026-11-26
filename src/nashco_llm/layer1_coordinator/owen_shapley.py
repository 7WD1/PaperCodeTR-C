"""Stage 2A: Fast sensitivity surrogate & two-level stratified Owen-Shapley (Section 3.3.2).

Implements:
- Eq. (16): \\gamma-core characteristic function v(S) = \\sum_{i in S} (C_i^{NonCoop} - C_i^{Coop}(S)).
- Eq. (17): Vectorized O(|A|) first-order sensitivity surrogate \\hat{v}(S) with one-step
  linearized outsider backfill \\Delta x_a^{br}(S).
- Eq. (18): Two-level stratified Monte Carlo Owen-Shapley estimator over MFD coalition
  structure B = {B_1, ..., B_K} achieving O(M^{-1/2}) Hoeffding concentration (Theorem 4).
"""

from __future__ import annotations

from typing import Callable, List, Sequence

import numpy as np


class StratifiedOwenEstimator:
    """Polynomial-time O(M |M| |A|) stratified two-level Owen-Shapley attribution engine."""

    def __init__(
        self,
        coalition_blocks: Sequence[Sequence[int]],
        sample_budget_m: int = 10_000,
        seed: int = 1,
    ) -> None:
        self.blocks: List[List[int]] = [list(b) for b in coalition_blocks if len(b) > 0]
        self.num_blocks = len(self.blocks)
        self.members: List[int] = sorted(i for b in self.blocks for i in b)
        self.num_members = len(self.members)
        self.sample_budget_m = sample_budget_m
        self.rng = np.random.default_rng(seed)

    @staticmethod
    def build_sensitivity_surrogate(
        x_noncoop_by_member: np.ndarray,
        delta_x_by_member: np.ndarray,
        link_cost_fn: Callable[[np.ndarray], np.ndarray],
        dc_dx_noncoop: np.ndarray,
        backfill_elasticity: float = 0.12,
    ) -> Callable[[np.ndarray], float]:
        """Construct vectorized Eq. (17) surrogate \\hat{v}(S) in O(|A|) per evaluation.

        Args:
            x_noncoop_by_member: Shape (|M|, |A|), non-cooperative link flows x_{a,i}^{NonCoop}.
            delta_x_by_member: Shape (|M|, |A|), cooperative flow shift \\Delta x_{a,i}.
            link_cost_fn: Vectorized \\tilde{c}_a(x_a) callable.
            dc_dx_noncoop: Link-cost Jacobian diagonal J_c at x^{NonCoop}.
            backfill_elasticity: Linearized outsider Newton backfill response factor.
        """
        x_nc = np.asarray(x_noncoop_by_member, dtype=float)
        dx = np.asarray(delta_x_by_member, dtype=float)
        x_total_nc = np.sum(x_nc, axis=0)
        c_nc = link_cost_fn(x_total_nc)
        member_nc_costs = x_nc @ c_nc  # (|M|,)

        def eval_v_hat(mask: np.ndarray) -> float:
            if not np.any(mask):
                return 0.0
            coop_shift = np.sum(dx[mask], axis=0)
            # One-step linearized outsider backfill \Delta x_a^{br}(S) for M \ S
            outsider_share = 1.0 - (float(np.sum(mask)) / float(len(mask)))
            slope_norm = dc_dx_noncoop / (np.max(dc_dx_noncoop) + 1e-9)
            backfill = -backfill_elasticity * outsider_share * slope_norm * coop_shift
            x_s = np.maximum(x_total_nc + coop_shift + backfill, 0.0)
            c_s = link_cost_fn(x_s)
            x_members_s = np.maximum(x_nc[mask] + dx[mask], 0.0)
            coop_cost_s = float(np.sum(x_members_s @ c_s))
            return float(np.sum(member_nc_costs[mask])) - coop_cost_s

        return eval_v_hat

    def estimate(self, v_oracle: Callable[[np.ndarray], float]) -> np.ndarray:
        """Estimate two-level Owen value \\psi^{Owen}(v, B) via Eq. (18) stratified draws.

        Draws outer regional permutations uniformly over K! and inner permutations within
        each regional union B_r uniformly over |B_r|!, guaranteeing exact unbiasedness
        E[\\hat{\\psi}_i^{Owen}] = \\psi_i^{Owen} (Theorem 4i) and O(M^{-1/2}) decay (Theorem 4ii).
        """
        owen_acc = np.zeros(self.num_members, dtype=float)
        member_to_pos = {m: idx for idx, m in enumerate(self.members)}

        for m_idx in range(self.sample_budget_m):
            # Stratified outer union ordering across K blocks
            outer_perm = self.rng.permutation(self.num_blocks)
            mask = np.zeros(self.num_members, dtype=bool)
            prev_val = 0.0

            for block_idx in outer_perm:
                inner_members = self.blocks[int(block_idx)]
                inner_perm = self.rng.permutation(inner_members)
                for member in inner_perm:
                    pos = member_to_pos[int(member)]
                    mask[pos] = True
                    new_val = v_oracle(mask)
                    owen_acc[pos] += new_val - prev_val
                    prev_val = new_val

        return owen_acc / float(self.sample_budget_m)

    @staticmethod
    def hoeffding_sample_bound(
        num_members: int,
        delta_v_max: float,
        epsilon_est: float,
        confidence_delta: float = 0.05,
    ) -> int:
        """Exact sample complexity from Theorem 4(ii), Eq. (27)."""
        m_req = (delta_v_max ** 2) * np.log(2.0 * num_members / confidence_delta) / (
            2.0 * (epsilon_est ** 2)
        )
        return int(np.ceil(m_req))
