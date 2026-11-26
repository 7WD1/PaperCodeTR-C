"""NashCo-LLM: Dual-Entropy Four-Layer Hierarchical Multi-Agent Control for DTA.

Reference:
    Jiang, W.-D., Wu, Z., & Long, J. (2026). When Language Meets Equilibrium:
    Decoupling Semantic Reasoning from Game-Theoretic Control in Dynamic Traffic
    Assignment. Transportation Research Part C: Emerging Technologies.
"""

from .config import NashCoConfig
from .pipeline import NashCoLLMPipeline
from .types import (
    ClearingResult,
    IncidentDispatch,
    IncidentPayload,
    LinkState,
    PerturbationTensor,
    UserClass,
)

__version__ = "1.0.0"

__all__ = [
    "NashCoConfig",
    "NashCoLLMPipeline",
    "ClearingResult",
    "IncidentDispatch",
    "IncidentPayload",
    "LinkState",
    "PerturbationTensor",
    "UserClass",
]
