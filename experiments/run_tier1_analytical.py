"""Reproduce Section 5.1 (Table 5a): Tier 1 Analytical Mechanism Verification.

Verifies Theorem 3(i) full Pigouvian internalization (\\eta_eff = 1.0) and Theorem 5
on the Braess paradox and two-fleet parallel networks with zero external dependencies.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nashco_llm.config import NashCoConfig
from nashco_llm.layer3_equilibrium.mirror_prox_vi import ODClassBlock
from nashco_llm.pipeline import NashCoLLMPipeline
from nashco_llm.simulators.bpr_analytical import (
    AnalyticalBPRSimulator,
    build_braess_network,
    build_two_fleet_network,
)
from nashco_llm.types import UserClass


def run_tier1(gamma0: float = 0.35, owen_samples: int = 500) -> None:
    cfg = NashCoConfig()
    cfg.horizon.num_periods = 3
    cfg.equilibrium.mirror_prox_gamma0 = gamma0
    cfg.cooperative.owen_samples_M = owen_samples
    cfg.regional.num_zones_K = 2

    print("=== Tier 1A: Braess Paradox Network (eta_eff = 1.0) ===")
    braess_net = build_braess_network(total_demand=4000.0)
    braess_sim = AnalyticalBPRSimulator(braess_net, cfg.equilibrium)
    braess_blocks = [
        ODClassBlock(od_id=0, user_class=UserClass.COOP, class_idx=0, demand=2000.0, path_indices=[0, 1, 2]),
        ODClassBlock(od_id=0, user_class=UserClass.COOP, class_idx=1, demand=2000.0, path_indices=[0, 1, 2]),
    ]
    pipe_b = NashCoLLMPipeline(
        topology=braess_net,
        simulator=braess_sim,
        blocks=braess_blocks,
        alpha_vot=np.array([1.0, 1.0]),
        config=cfg,
        use_full_tier1_eta=True,
    )
    trace_b = pipe_b.run_horizon(enable_msc=True, enable_llm=False)
    last_b = trace_b.periods[-1]
    print(
        f"  Converged in {last_b.vi_iterations} iters | TSTT={last_b.tstt:.2f} "
        f"| core_eps={last_b.clearing.core_violation:.5f} | budget_res={last_b.clearing.budget_residual:.2e}"
    )

    print("\n=== Tier 1B: Two-Fleet Parallel Network (Heterogeneous Atomic VOT) ===")
    fleet_net = build_two_fleet_network(total_demand=6000.0)
    fleet_sim = AnalyticalBPRSimulator(fleet_net, cfg.equilibrium)
    fleet_blocks = [
        ODClassBlock(
            od_id=0, user_class=UserClass.COOP, class_idx=0, demand=3000.0,
            path_indices=[0, 1], is_atomic_fleet=True, fleet_member_id=0,
        ),
        ODClassBlock(
            od_id=0, user_class=UserClass.COOP, class_idx=1, demand=3000.0,
            path_indices=[0, 1], is_atomic_fleet=True, fleet_member_id=1,
        ),
    ]
    pipe_f = NashCoLLMPipeline(
        topology=fleet_net,
        simulator=fleet_sim,
        blocks=fleet_blocks,
        alpha_vot=np.array([1.0, 1.35]),  # Heterogeneous VOT per Theorem 3(i)
        config=cfg,
        use_full_tier1_eta=True,
    )
    trace_f = pipe_f.run_horizon(enable_msc=True, enable_llm=False)
    last_f = trace_f.periods[-1]
    print(
        f"  Converged in {last_f.vi_iterations} iters | TSTT={last_f.tstt:.2f} "
        f"| core_eps={last_f.clearing.core_violation:.5f} | budget_res={last_f.clearing.budget_residual:.2e}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Tier 1 analytical verification (Section 5.1).")
    parser.add_argument("--gamma0", type=float, default=0.35, help="Initial Mirror-Prox step size.")
    parser.add_argument("--owen-samples", type=int, default=500, help="Stratified Owen draws M.")
    args = parser.parse_args()
    run_tier1(gamma0=args.gamma0, owen_samples=args.owen_samples)
