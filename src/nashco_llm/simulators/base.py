"""Abstract dynamic network loading (DNL) simulator interface & TNTP network container."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from ..types import LinkState, PerturbationTensor


@dataclass
class NetworkTopology:
    """Directed transportation graph G=(V, A) with path-link incidence matrix."""

    name: str
    num_nodes: int
    link_endpoints: List[Tuple[int, int]]
    free_flow_times: np.ndarray
    capacities: np.ndarray
    free_flow_speeds_kmh: np.ndarray
    path_link_incidence: np.ndarray   # Shape (|P|, |A|)
    od_paths: Dict[int, List[int]]    # od_id -> list of path indices
    od_demands: Dict[int, float]      # od_id -> total demand


class TrafficSimulator(ABC):
    """Unified DNL interface \\Lambda(f; \\Xi) across Analytical BPR, UXsim, DTALite, SUMO."""

    def __init__(self, topology: NetworkTopology) -> None:
        self.topology = topology

    @abstractmethod
    def load_flows(
        self,
        link_flows: np.ndarray,
        class_link_flows: np.ndarray,
        xi: PerturbationTensor,
    ) -> LinkState:
        """Execute dynamic network loading and return realized LinkState."""
