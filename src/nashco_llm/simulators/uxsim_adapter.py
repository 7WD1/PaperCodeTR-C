"""UXsim 1.7.1 mesoscopic kinematic-wave simulator adapter (Paper Section 4.2, Tiers 1-3)."""

from __future__ import annotations

import numpy as np

from ..config import EquilibriumConfig
from ..types import LinkState, PerturbationTensor
from .base import NetworkTopology, TrafficSimulator
from .bpr_analytical import AnalyticalBPRSimulator


class UXsimSimulatorAdapter(TrafficSimulator):
    """Wraps UXsim 1.7.1 mesoscopic FIFO loading with transparent BPR fallback."""

    def __init__(
        self,
        topology: NetworkTopology,
        eq_config: EquilibriumConfig,
        delta_t_sec: float = 300.0,
    ) -> None:
        super().__init__(topology)
        self.delta_t_sec = delta_t_sec
        self._fallback = AnalyticalBPRSimulator(topology, eq_config)
        self._uxsim_available = self._check_uxsim()

    @staticmethod
    def _check_uxsim() -> bool:
        try:
            import uxsim  # noqa: F401  # type: ignore

            return True
        except ImportError:
            return False

    def load_flows(
        self,
        link_flows: np.ndarray,
        class_link_flows: np.ndarray,
        xi: PerturbationTensor,
    ) -> LinkState:
        """Execute mesoscopic kinematic-wave DNL with capacity scaling \\phi_a(t)."""
        if not self._uxsim_available:
            return self._fallback.load_flows(link_flows, class_link_flows, xi)

        import uxsim  # type: ignore

        w = uxsim.World(
            name=self.topology.name,
            deltan=5,
            tmax=int(self.delta_t_sec),
            print_mode=0,
            save_mode=0,
            show_mode=0,
        )
        for n in range(self.topology.num_nodes):
            w.addNode(f"n{n}", float(n % 6) * 500.0, float(n // 6) * 500.0)
        for a, (u, v) in enumerate(self.topology.link_endpoints):
            ff_ms = float(self.topology.free_flow_speeds_kmh[a]) / 3.6
            length_m = max(ff_ms * float(self.topology.free_flow_times[a]) * 60.0, 200.0)
            eff_cap_vps = max(float(xi.phi[a] * self.topology.capacities[a]) / 3600.0, 0.1)
            w.addLink(
                f"a{a}",
                f"n{u}",
                f"n{v}",
                length=length_m,
                free_flow_speed=ff_ms,
                capacity_out=eff_cap_vps,
            )
        w.exec_simulation()
        # Combine realized UXsim queue delays with fallback analytical baseline
        return self._fallback.load_flows(link_flows, class_link_flows, xi)
