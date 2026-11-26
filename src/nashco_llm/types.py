"""Core data structures shared across Layers 1-4 of NashCo-LLM (Table 2 & Table 3)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import numpy as np


class UserClass(str, Enum):
    """Four coexisting behavioral user classes N = N_HDV U N_UE U N_FO U N_Coop."""

    HDV = "hdv"          # Human-driven vehicles (SUE, \tau_m = \theta_HDV^{-1})
    UE = "ue"            # Non-cooperative private CAVs (UE, \tau_m = \tau)
    FO = "fo"            # Non-cooperative commercial fleets (Cournot-Nash FO)
    COOP = "coop"        # Cooperative alliance members M = N_Coop


class PriorityFlag(str, Enum):
    """Gate 4 authorization flag for emergency corridors and road bans."""

    NONE = "none"
    EMERG = "emerg"
    BAN = "ban"


@dataclass
class IncidentDispatch:
    """Raw unstructured dispatch text entering Layer 4 with authentication metadata."""

    dispatch_id: str
    period: int
    raw_text: str
    source: str = "chp_official"
    is_official_agency: bool = True
    reported_milepost: Optional[str] = None
    geofence_zones: List[int] = field(default_factory=list)


@dataclass
class IncidentPayload:
    """Strict 7-field structured JSON Schema payload (Paper Table 3)."""

    link_ids: List[int]
    severity: int
    phi: float
    duration: int
    confidence: float
    priority_flag: str
    rationale: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "link_ids": self.link_ids,
            "severity": self.severity,
            "phi": self.phi,
            "duration": self.duration,
            "confidence": self.confidence,
            "priority_flag": self.priority_flag,
            "rationale": self.rationale,
        }


@dataclass
class PerturbationTensor:
    """Verified compact perturbation tensor \\Xi(t) \\in D_\\Xi (Eq. 3)."""

    phi: np.ndarray          # Shape (|A|,), capacity retention \phi_a \in [\phi_ban, 1]
    hazard_risk: np.ndarray  # Shape (|A|,), confidence-weighted risk \tilde{r}_a \in [0, r_max]
    zeta_bias: np.ndarray    # Shape (|A|, |N|), class-specific priority/preference bias \zeta_{a,m}

    @classmethod
    def nominal(cls, num_links: int, num_classes: int = 4) -> "PerturbationTensor":
        return cls(
            phi=np.ones(num_links, dtype=float),
            hazard_risk=np.zeros(num_links, dtype=float),
            zeta_bias=np.zeros((num_links, num_classes), dtype=float),
        )


@dataclass
class LinkState:
    """Physical dynamic network loading (DNL) state returned by \\Lambda(f)."""

    flows: np.ndarray              # Shape (|A|,), total link volumes x_a(t)
    class_flows: np.ndarray        # Shape (|A|, |N|), per-class volumes x_{a,m}(t)
    travel_times: np.ndarray       # Shape (|A|,), physical travel time \ell_a(t)
    speeds_kmh: np.ndarray         # Shape (|A|,), realized link speeds v_a(t)
    predicted_speeds_kmh: np.ndarray  # Shape (|A|,), one-step predicted speeds \hat{v}_a(t)
    free_flow_speeds_kmh: np.ndarray  # Shape (|A|,), nominal free-flow speeds v_a^ff


@dataclass
class ClearingResult:
    """Stage 2 cooperative clearing output from Owen-Shapley + penalized \\varepsilon-core QP."""

    grand_surplus: float               # v(M)
    owen_values: np.ndarray            # \psi^{Owen}
    allocated_surplus: np.ndarray      # y^*
    credit_rebates: np.ndarray         # b^*(t) from Eq. (22)
    core_violation: float              # \varepsilon^* (normalized)
    budget_residual: float             # |\sum b_i^* - \sum \pi x|
    pareto_improvement_ratio: float    # PIR (%)
    cold_start_bypass: bool = False
