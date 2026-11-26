"""Built-in analytical BPR / point-queue DNL simulator & Tier 1 benchmark builders."""

from __future__ import annotations

import numpy as np

from ..config import EquilibriumConfig
from ..layer3_equilibrium.traffic_cost import SemanticTrafficCostModel
from ..types import LinkState, PerturbationTensor
from .base import NetworkTopology, TrafficSimulator


class AnalyticalBPRSimulator(TrafficSimulator):
    """Self-contained BPR DNL engine for Tier 1 exact verification & offline CI testing."""

    def __init__(self, topology: NetworkTopology, eq_config: EquilibriumConfig) -> None:
        super().__init__(topology)
        self.cost_model = SemanticTrafficCostModel(
            topology.free_flow_times, topology.capacities, eq_config
        )
        self._last_speeds = topology.free_flow_speeds_kmh.copy()

    def load_flows(
        self,
        link_flows: np.ndarray,
        class_link_flows: np.ndarray,
        xi: PerturbationTensor,
    ) -> LinkState:
        tt = self.cost_model.link_travel_time(link_flows, xi.phi)
        pred_tt = self.cost_model.link_travel_time(link_flows, np.ones_like(xi.phi))
        speeds = self.topology.free_flow_speeds_kmh * (self.topology.free_flow_times / np.maximum(tt, 1e-6))
        pred_speeds = self.topology.free_flow_speeds_kmh * (
            self.topology.free_flow_times / np.maximum(pred_tt, 1e-6)
        )
        self._last_speeds = speeds
        return LinkState(
            flows=link_flows.copy(),
            class_flows=class_link_flows.copy(),
            travel_times=tt,
            speeds_kmh=speeds,
            predicted_speeds_kmh=pred_speeds,
            free_flow_speeds_kmh=self.topology.free_flow_speeds_kmh.copy(),
        )


def build_braess_network(total_demand: float = 4000.0) -> NetworkTopology:
    """Classic Tier 1 Braess paradox network (4 nodes, 5 links, 1 OD, UE=2.0 / SO=1.5)."""
    # Links: 0:(0->1, linear), 1:(1->3, const), 2:(0->2, const), 3:(2->3, linear), 4:(1->2, shortcut)
    # Paths: p0=[0,1], p1=[2,3], p2=[0,4,3]
    incidence = np.array(
        [
            [1, 1, 0, 0, 0],
            [0, 0, 1, 1, 0],
            [1, 0, 0, 1, 1],
        ],
        dtype=float,
    )
    return NetworkTopology(
        name="Braess paradox",
        num_nodes=4,
        link_endpoints=[(0, 1), (1, 3), (0, 2), (2, 3), (1, 2)],
        free_flow_times=np.array([0.4, 1.0, 1.0, 0.4, 0.05], dtype=float),
        capacities=np.array([2000.0, 8000.0, 8000.0, 2000.0, 8000.0], dtype=float),
        free_flow_speeds_kmh=np.array([70.0, 80.0, 80.0, 70.0, 90.0], dtype=float),
        path_link_incidence=incidence,
        od_paths={0: [0, 1, 2]},
        od_demands={0: total_demand},
    )


def build_two_fleet_network(total_demand: float = 6000.0) -> NetworkTopology:
    """Tier 1 two-fleet parallel bottleneck network (4 nodes, 2 links, 1 OD)."""
    incidence = np.array([[1, 0], [0, 1]], dtype=float)
    return NetworkTopology(
        name="Two-fleet parallel",
        num_nodes=4,
        link_endpoints=[(0, 1), (0, 2)],
        free_flow_times=np.array([15.0, 17.5], dtype=float),
        capacities=np.array([2800.0, 3600.0], dtype=float),
        free_flow_speeds_kmh=np.array([85.0, 75.0], dtype=float),
        path_link_incidence=incidence,
        od_paths={0: [0, 1]},
        od_demands={0: total_demand},
    )
