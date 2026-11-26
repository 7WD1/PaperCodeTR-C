"""Typed configuration dataclasses mirroring Paper Table 2 and Table 4(b)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Tuple

import yaml


@dataclass
class HorizonConfig:
    delta_t_min: float = 5.0
    num_periods: int = 20


@dataclass
class RegionalConfig:
    num_zones_K: int = 8
    trigger_threshold: float = 0.12
    top_q_links: int = 3
    v_min_kmh: float = 5.0


@dataclass
class EquilibriumConfig:
    bpr_b: float = 0.15
    bpr_p: float = 4.0
    beta_safe_ratio: float = 0.15
    risk_density_coeff: float = 0.15
    theta_hdv: float = 1.0
    entropy_tau: float = 0.08
    mirror_prox_gamma0: float = 0.35
    mirror_prox_kappa: float = 0.60
    vi_tolerance: float = 1.0e-6
    vi_edge_tolerance: float = 1.0e-5
    max_vi_iters: int = 200


@dataclass
class SemanticConfig:
    provider: str = "mock"
    model_name: str = "gpt-4o-2024-11-20"
    temperature: float = 0.2
    top_p: float = 0.9
    max_tokens: int = 512
    rag_embedding_model: str = "text-embedding-3-small"
    rag_top_k: int = 8
    rag_cosine_threshold: float = 0.78
    max_reflection_retries: int = 1
    phi_min_routine: float = 0.20
    phi_ban_closure: float = 0.05
    ramp_recovery_cap: float = 0.25
    r_max: float = 1.0
    zeta_max: float = 2.0
    calib_interval_steps: int = 3
    calib_lambda: float = 0.25
    alpha_bounds: Tuple[float, float] = (0.5, 3.0)


@dataclass
class CooperativeConfig:
    eta_eff_base: float = 0.05
    eta_eff_full: float = 1.00
    adaptive_eta_enabled: bool = True
    adaptive_beta: float = 1.35
    eta_bounds: Tuple[float, float] = (0.05, 0.25)
    compliance_defection_gain: float = 1.5
    compliance_min: float = 0.60
    owen_samples_M: int = 10_000
    num_critical_cuts: int = 64
    qp_smoothing_rel: float = 1.0e-3
    qp_penalty_mu: float = 10.0
    qp_solver: str = "scipy"
    qp_tolerance: float = 1.0e-8
    initial_wallet_endowment: float = 100.0


@dataclass
class NashCoConfig:
    """Complete configuration bundle for the four-layer NashCo-LLM controller."""

    horizon: HorizonConfig = field(default_factory=HorizonConfig)
    regional: RegionalConfig = field(default_factory=RegionalConfig)
    equilibrium: EquilibriumConfig = field(default_factory=EquilibriumConfig)
    semantic: SemanticConfig = field(default_factory=SemanticConfig)
    cooperative: CooperativeConfig = field(default_factory=CooperativeConfig)
    seed: int = 1

    @classmethod
    def from_yaml(cls, path: str | Path) -> "NashCoConfig":
        with Path(path).open("r", encoding="utf-8") as fh:
            raw: Dict[str, Any] = yaml.safe_load(fh)
        sem_raw = dict(raw.get("layer4_semantic", {}))
        if "alpha_bounds" in sem_raw:
            sem_raw["alpha_bounds"] = tuple(sem_raw["alpha_bounds"])
        coop_raw = dict(raw.get("layer1_cooperative", {}))
        if "eta_bounds" in coop_raw:
            coop_raw["eta_bounds"] = tuple(coop_raw["eta_bounds"])
        return cls(
            horizon=HorizonConfig(**raw.get("horizon", {})),
            regional=RegionalConfig(**raw.get("layer2_regional", {})),
            equilibrium=EquilibriumConfig(**raw.get("layer3_equilibrium", {})),
            semantic=SemanticConfig(**sem_raw),
            cooperative=CooperativeConfig(**coop_raw),
        )
