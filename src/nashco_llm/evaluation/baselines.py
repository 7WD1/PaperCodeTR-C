"""Registry of all 12 published 2025-2026 baselines & reference bounds (Section 4.3)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class BaselineSpec:
    """Metadata and adaptation protocol for each comparative benchmark method."""

    key: str
    citation: str
    family: str
    uses_text_frontend: bool
    native_query_cadence_sf: int
    description: str


BASELINE_REGISTRY: Dict[str, BaselineSpec] = {
    # Reference bounds
    "dso_bound": BaselineSpec(
        "dso_bound", "Reference Bound", "reference", False, 0,
        "Rolling dynamic system optimum with perfect disruption foresight.",
    ),
    "msa_due": BaselineSpec(
        "msa_due", "Reference Bound", "reference", False, 0,
        "Method of Successive Averages dynamic user equilibrium denominator.",
    ),
    # Cooperative game & mobility-credit baselines
    "han2026": BaselineSpec(
        "han2026", "Han et al. (2026), TR-E", "cooperative_or", True, 486,
        "Online rolling adaptation of two-echelon fast Shapley sharing.",
    ),
    "rodriguez2026": BaselineSpec(
        "rodriguez2026", "Rodriguez-Pereira et al. (2026), TR-E", "cooperative_or", True, 312,
        "Rolling nucleolus cost-sharing allocation (best of Shapley, nucleolus, EPM).",
    ),
    "rao2025": BaselineSpec(
        "rao2025", "Rao et al. (2025), TR-E", "cooperative_or", True, 402,
        "Online Shapley horizontal collaborative routing cost sharing.",
    ),
    "ye2026": BaselineSpec(
        "ye2026", "Ye et al. (2026), TR-E", "cooperative_or", True, 96,
        "Link tradable mobility-credit pricing without core-projection clearing.",
    ),
    # Physics-guided & equilibrium DTA baselines
    "perl2026": BaselineSpec(
        "perl2026", "Shi et al. (2026), TR-C", "physics_dta", False, 0,
        "Iterative physics-enhanced residual learning for context-aware assignment.",
    ),
    "transrl2025": BaselineSpec(
        "transrl2025", "Ke et al. (2025), TR-C", "physics_dta", False, 0,
        "Differentiable physics teacher coupled to RL student controller.",
    ),
    "morandi2026": BaselineSpec(
        "morandi2026", "Morandi et al. (2026), TR-C", "physics_dta", False, 0,
        "Risk-controlled real-time traffic rerouting under structured disruptions.",
    ),
    "smtap2026": BaselineSpec(
        "smtap2026", "Tang et al. (2026), TR-E", "physics_dta", False, 0,
        "Safety-aware multi-player VI across HDVs, CAVs, and commercial fleets.",
    ),
    # Pure LLM routing agents
    "llm_vd2026": BaselineSpec(
        "llm_vd2026", "Shi & Zhen (2026), TR-E", "llm_agent", True, 24,
        "LLM-based heuristic synthesis with dual self-debugging adapted to rerouting.",
    ),
    "dual_agent2026": BaselineSpec(
        "dual_agent2026", "Liu et al. (2026b), TR-C", "llm_agent", True, 2112,
        "Traveler LLM paired with online behavioral calibrator LLM.",
    ),
    "llmtraveler2025": BaselineSpec(
        "llmtraveler2025", "Wang et al. (2025a), TR-C", "llm_agent", True, 20000,
        "EWMATT memory and persona-driven day-to-day LLM route choice agent.",
    ),
    "chatplanner2026": BaselineSpec(
        "chatplanner2026", "Yang et al. (2026), TR-C", "llm_agent", True, 20000,
        "LangGraph + RAG routing agent adapted to rolling road assignment.",
    ),
}
