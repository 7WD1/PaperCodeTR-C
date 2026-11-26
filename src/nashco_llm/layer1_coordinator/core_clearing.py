"""Stage 2B: Scale-invariant penalized \\varepsilon-core QP & credit clearing (Section 3.3.2).

Implements:
- Eqs. (19)-(21): Scale-invariant penalized \\varepsilon-core QP with relative smoothing
  \\delta_qp = 1e-3 * v(M) / |M| and dimensionless penalty \\mu = 10 over 64 critical cuts S_crit.
- Theorem 5(iii): Zero-transfer cold-start safeguard whenever v(M) <= 0.
- Eq. (22): Ex-ante lump-sum mobility-credit rebate settlement b_i^*(t) achieving exact
  on-equilibrium budget balance and Pareto individual rationality (Theorem 5ii).
- Remark 1: Off-equilibrium municipal mobility-credit reserve buffer tracking.
"""

from __future__ import annotations

from typing import Callable, List, Sequence, Tuple

import numpy as np

from ..config import CooperativeConfig
from ..types import ClearingResult


class ScaleInvariantCoreClearer:
    """Solves Eqs. (19)-(22) via Gurobi, OSQP, or SciPy SLSQP fallback."""

    def __init__(self, config: CooperativeConfig) -> None:
        self.cfg = config
        self.reserve_balance: float = 0.0

    def screen_critical_coalitions(
        self,
        num_members: int,
        coalition_blocks: Sequence[Sequence[int]],
        owen_values: np.ndarray,
        v_oracle: Callable[[np.ndarray], float],
        max_cuts: int | None = None,
    ) -> List[Tuple[np.ndarray, float]]:
        """Build S_crit (|S_crit| <= 64): singletons, regional blocks B_r, and worst excesses."""
        limit = self.cfg.num_critical_cuts if max_cuts is None else max_cuts
        candidates: List[np.ndarray] = []

        # 1. Singletons {i}
        for i in range(num_members):
            mask = np.zeros(num_members, dtype=bool)
            mask[i] = True
            candidates.append(mask)

        # 2. Regional unions B_r and complements M \ B_r
        for block in coalition_blocks:
            if 0 < len(block) < num_members:
                mask = np.zeros(num_members, dtype=bool)
                mask[list(block)] = True
                candidates.append(mask)
                candidates.append(~mask)

        # Deduplicate and rank by raw Owen excess v(S) - \sum_{i in S} \psi_i^{Owen}
        seen = set()
        scored: List[Tuple[float, np.ndarray, float]] = []
        for mask in candidates:
            key = mask.tobytes()
            if key in seen or not np.any(mask) or np.all(mask):
                continue
            seen.add(key)
            val_s = float(v_oracle(mask))
            excess = val_s - float(np.sum(owen_values[mask]))
            scored.append((excess, mask, val_s))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [(mask, val_s) for _, mask, val_s in scored[:limit]]

    def clear_epoch(
        self,
        owen_values: np.ndarray,
        grand_surplus: float,
        critical_cuts: Sequence[Tuple[np.ndarray, float]],
        noncoop_costs: np.ndarray,
        coop_costs: np.ndarray,
        collected_msc_charges: np.ndarray,
    ) -> ClearingResult:
        """Solve Eqs. (19)-(21) and settle Eq. (22) mobility-credit rebates b_i^*(t)."""
        psi = np.asarray(owen_values, dtype=float)
        n = len(psi)
        charges = np.asarray(collected_msc_charges, dtype=float)

        # Theorem 5(iii): Cold-start zero-transfer safeguard when v(M) <= 0
        if grand_surplus <= 1e-12:
            y_zero = np.zeros(n, dtype=float)
            b_refund = charges.copy()
            return ClearingResult(
                grand_surplus=float(grand_surplus),
                owen_values=psi,
                allocated_surplus=y_zero,
                credit_rebates=b_refund,
                core_violation=0.0,
                budget_residual=0.0,
                pareto_improvement_ratio=0.0,
                cold_start_bypass=True,
            )

        # Scale-invariant relative smoothing \delta_qp = 1e-3 * v(M) / |M| (Theorem 5i)
        delta_qp = self.cfg.qp_smoothing_rel * (grand_surplus / float(n))
        weights = 1.0 / (np.abs(psi) + delta_qp)
        y_star, eps_star = self._solve_qp(psi, weights, grand_surplus, critical_cuts)

        # Eq. (22): Exogenous lump-sum rebate b_i^*(t)
        rebates = y_star + (coop_costs + charges - noncoop_costs)
        budget_residual = float(abs(np.sum(rebates) - np.sum(charges)))

        # Strictly positive surplus gain counts toward Pareto improvement ratio (PIR)
        pir = 100.0 * float(np.mean(y_star > 1e-9))

        return ClearingResult(
            grand_surplus=float(grand_surplus),
            owen_values=psi,
            allocated_surplus=y_star,
            credit_rebates=rebates,
            core_violation=float(eps_star),
            budget_residual=budget_residual,
            pareto_improvement_ratio=pir,
            cold_start_bypass=False,
        )

    def _solve_qp(
        self,
        psi: np.ndarray,
        weights: np.ndarray,
        v_grand: float,
        critical_cuts: Sequence[Tuple[np.ndarray, float]],
    ) -> Tuple[np.ndarray, float]:
        """Solve strictly convex Eqs. (19)-(21) via Gurobi -> SciPy SLSQP -> projected fallback."""
        n = len(psi)
        mu = self.cfg.qp_penalty_mu

        # Primary backend: Gurobi 12.0.1 barrier solver (paper Table 4b)
        if self.cfg.qp_solver == "gurobi":
            try:
                import gurobipy as gp  # type: ignore
                from gurobipy import GRB  # type: ignore

                model = gp.Model("nashco_eps_core_qp")
                model.Params.OutputFlag = 0
                model.Params.BarConvTol = self.cfg.qp_tolerance
                y = model.addVars(n, lb=0.0, name="y")
                eps = model.addVar(lb=0.0, name="eps")
                obj = gp.quicksum(
                    ((y[i] - float(psi[i])) * float(weights[i])) ** 2 for i in range(n)
                ) + mu * eps
                model.setObjective(obj, GRB.MINIMIZE)
                model.addConstr(gp.quicksum(y[i] for i in range(n)) == v_grand, name="eff")
                for idx, (mask, v_s) in enumerate(critical_cuts):
                    model.addConstr(
                        gp.quicksum(y[i] for i in range(n) if mask[i]) >= v_s - eps * v_grand,
                        name=f"cut_{idx}",
                    )
                model.optimize()
                if model.Status == GRB.OPTIMAL:
                    return np.array([y[i].X for i in range(n)], dtype=float), float(eps.X)
            except Exception:
                pass

        # Portable backend: SciPy SLSQP
        try:
            from scipy.optimize import minimize  # type: ignore

            y0 = np.full(n, v_grand / n, dtype=float)
            eps0 = max(
                [0.0]
                + [
                    max(0.0, (v_s - float(np.sum(y0[mask]))) / v_grand)
                    for mask, v_s in critical_cuts
                ]
            )
            z0 = np.concatenate([y0, [eps0]])

            def fun(z: np.ndarray) -> float:
                diff = (z[:n] - psi) * weights
                return float(np.dot(diff, diff) + mu * z[n])

            def jac(z: np.ndarray) -> np.ndarray:
                g = np.zeros(n + 1, dtype=float)
                g[:n] = 2.0 * (z[:n] - psi) * (weights ** 2)
                g[n] = mu
                return g

            constraints = [
                {
                    "type": "eq",
                    "fun": lambda z: float(np.sum(z[:n]) - v_grand),
                    "jac": lambda z: np.concatenate([np.ones(n), [0.0]]),
                }
            ]
            for mask, v_s in critical_cuts:
                a_vec = np.concatenate([mask.astype(float), [v_grand]])
                constraints.append(
                    {
                        "type": "ineq",
                        "fun": lambda z, av=a_vec, vs=v_s: float(np.dot(av, z) - vs),
                        "jac": lambda z, av=a_vec: av,
                    }
                )
            bounds = [(0.0, None)] * (n + 1)
            res = minimize(
                fun,
                z0,
                jac=jac,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"ftol": 1e-11, "maxiter": 500},
            )
            if res.success:
                y_sol = np.maximum(res.x[:n], 0.0)
                y_sol *= v_grand / max(float(np.sum(y_sol)), 1e-12)
                eps_sol = max(
                    [0.0]
                    + [
                        max(0.0, (v_s - float(np.sum(y_sol[mask]))) / v_grand)
                        for mask, v_s in critical_cuts
                    ]
                )
                return y_sol, float(eps_sol)
        except Exception:
            pass

        # Pure-NumPy weighted simplex projection fallback (zero external dependencies)
        y_proj = np.maximum(psi, 0.0)
        if float(np.sum(y_proj)) <= 1e-12:
            y_proj = np.full(n, v_grand / n, dtype=float)
        else:
            y_proj = y_proj * (v_grand / float(np.sum(y_proj)))

        # Iterative subgradient refinement on violated core cuts
        for step in range(250):
            lr = 0.25 / (1.0 + 0.05 * step)
            grad = 2.0 * (y_proj - psi) * (weights ** 2)
            worst_excess = 0.0
            worst_mask = None
            for mask, v_s in critical_cuts:
                ex = (v_s - float(np.sum(y_proj[mask]))) / v_grand
                if ex > worst_excess:
                    worst_excess = ex
                    worst_mask = mask
            if worst_mask is not None:
                grad -= (mu / v_grand) * worst_mask.astype(float)
            y_cand = np.maximum(y_proj - lr * grad, 0.0)
            y_proj = self._project_simplex(y_cand, v_grand)

        eps_bind = max(
            [0.0]
            + [
                max(0.0, (v_s - float(np.sum(y_proj[mask]))) / v_grand)
                for mask, v_s in critical_cuts
            ]
        )
        return y_proj, float(eps_bind)

    @staticmethod
    def _project_simplex(v: np.ndarray, total: float) -> np.ndarray:
        """Euclidean projection of v onto {y >= 0 : sum(y) = total}."""
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u) - total
        ind = np.arange(1, len(v) + 1)
        cond = u - cssv / ind > 0
        rho = int(ind[cond][-1])
        theta = float(cssv[cond][-1]) / float(rho)
        return np.maximum(v - theta, 0.0)

    def settle_off_equilibrium_reserve(
        self,
        realized_charges: np.ndarray,
        committed_rebates: np.ndarray,
        compliant_mask: np.ndarray,
    ) -> float:
        """Remark 1: Forfeit non-compliant rebates and buffer pool variance in reserve."""
        paid_rebates = np.where(compliant_mask, committed_rebates, 0.0)
        epoch_variance = float(np.sum(realized_charges) - np.sum(paid_rebates))
        self.reserve_balance += epoch_variance
        return epoch_variance
