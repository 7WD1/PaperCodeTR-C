"""Multi-fidelity traffic simulator interfaces (Paper Section 4.2)."""

from .base import NetworkTopology, TrafficSimulator
from .bpr_analytical import AnalyticalBPRSimulator, build_braess_network, build_two_fleet_network
from .dtalite_adapter import DTALiteSimulatorAdapter
from .sumo_traci_adapter import SUMOTraCISimulatorAdapter
from .uxsim_adapter import UXsimSimulatorAdapter

__all__ = [
    "NetworkTopology",
    "TrafficSimulator",
    "AnalyticalBPRSimulator",
    "build_braess_network",
    "build_two_fleet_network",
    "UXsimSimulatorAdapter",
    "DTALiteSimulatorAdapter",
    "SUMOTraCISimulatorAdapter",
]
