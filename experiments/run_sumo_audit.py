"""Reproduce Section 5.4 (Table 10 & Fig. 9): Calibrated Microscopic SUMO Audit.

Evaluates TET conflict exposure (1.5 s TTC threshold), queue spillback (veh-h),
signalized intersection delay (s), and relative travel time ratio on the calibrated
5% demand-scaled Sioux Falls subarea network (Krauss car-following, LC2013 lane-changing).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nashco_llm.config import NashCoConfig
from nashco_llm.simulators.bpr_analytical import build_two_fleet_network
from nashco_llm.simulators.sumo_traci_adapter import SUMOTraCISimulatorAdapter
from nashco_llm.types import PerturbationTensor


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SUMO 1.21.0 microscopic safety & spillback audit.")
    parser.add_argument("--sumo-cfg", type=str, default=None, help="Path to calibrated Sioux Falls .sumocfg.")
    parser.add_argument("--ttc", type=float, default=1.5, help="TTC threshold (s) for TET conflict metric.")
    args = parser.parse_args()

    cfg = NashCoConfig()
    topo = build_two_fleet_network()
    adapter = SUMOTraCISimulatorAdapter(
        topology=topo,
        eq_config=cfg.equilibrium,
        sumo_cfg_path=args.sumo_cfg,
        ttc_threshold_sec=args.ttc,
        demand_scale=0.05,
    )
    xi = PerturbationTensor.nominal(len(topo.capacities), num_classes=4)
    state = adapter.load_flows(np.array([2900.0, 3100.0]), np.full((2, 4), 750.0), xi)
    audit = adapter.run_microscopic_audit(state, xi)
    print(f"[SUMO Audit] TTC={args.ttc}s | {audit}")


if __name__ == "__main__":
    main()
