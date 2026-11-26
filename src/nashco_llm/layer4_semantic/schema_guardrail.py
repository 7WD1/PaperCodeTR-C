"""Deterministic four-gate physical guardrail & 7-field JSON Schema (Section 3.2.1 & Table 3).

Guarantees that 100% of perturbation tensors entering DNL lie inside the compact box:
    D_\\Xi = [\\phi_ban, 1]^{|A|} x [0, r_max]^{|A|} x [-\\zeta_max, \\zeta_max]^{|A| x |N|}   (Eq. 3).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set

import numpy as np

from ..config import SemanticConfig
from ..types import IncidentDispatch, IncidentPayload, PerturbationTensor, UserClass

# Structural 7-field JSON Schema specification (Paper Table 3; physical bounds checked in Gates 1-4)
INCIDENT_JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": [
        "link_ids",
        "severity",
        "phi",
        "duration",
        "confidence",
        "priority_flag",
        "rationale",
    ],
    "additionalProperties": False,
    "properties": {
        "link_ids": {"type": "array", "items": {"type": "integer"}, "minItems": 1},
        "severity": {"type": "integer", "minimum": 1, "maximum": 4},
        "phi": {"type": "number"},
        "duration": {"type": "integer", "minimum": 1, "maximum": 24},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "priority_flag": {"type": "string", "enum": ["none", "emerg", "ban"]},
        "rationale": {"type": "string", "minLength": 1},
    },
}

SEVERITY_RISK_MAP: Dict[int, float] = {1: 0.15, 2: 0.40, 3: 0.70, 4: 1.00}


@dataclass
class GuardrailAudit:
    """Diagnostic audit trace produced by the four sequential guardrail gates."""

    passed: bool
    violations: List[str] = field(default_factory=list)
    payload: Optional[IncidentPayload] = None


class FourGatePhysicalGuardrail:
    """Sequential four-gate deterministic verifier before solver injection."""

    def __init__(self, config: SemanticConfig) -> None:
        self.cfg = config

    def validate(
        self,
        raw_json: Dict[str, Any],
        allowed_links: Set[int],
        prev_phi: np.ndarray,
        dispatch: IncidentDispatch,
        is_incident_onset: bool = True,
    ) -> GuardrailAudit:
        """Evaluate Schema + Gates 1-4 sequentially and return exact error trace."""
        violations: List[str] = []

        # Schema structural validation (Table 3) with pure-Python fallback if jsonschema is absent
        try:
            import jsonschema  # type: ignore

            try:
                jsonschema.validate(instance=raw_json, schema=INCIDENT_JSON_SCHEMA)
            except Exception as exc:
                violations.append(f"SchemaError: {exc}")
                return GuardrailAudit(passed=False, violations=violations)
        except ImportError:
            req = set(INCIDENT_JSON_SCHEMA["required"])
            if not isinstance(raw_json, dict) or set(raw_json.keys()) != req:
                violations.append(f"SchemaError: expected exact 7 fields {sorted(req)}.")
                return GuardrailAudit(passed=False, violations=violations)
            if not isinstance(raw_json["link_ids"], list) or len(raw_json["link_ids"]) < 1:
                violations.append("SchemaError: link_ids must be a non-empty list[int].")
                return GuardrailAudit(passed=False, violations=violations)
            if raw_json["severity"] not in (1, 2, 3, 4):
                violations.append("SchemaError: severity must be int in {1,2,3,4}.")
                return GuardrailAudit(passed=False, violations=violations)
            if not (1 <= int(raw_json["duration"]) <= 24):
                violations.append("SchemaError: duration must be int in [1,24].")
                return GuardrailAudit(passed=False, violations=violations)
            if not (0.0 <= float(raw_json["confidence"]) <= 1.0):
                violations.append("SchemaError: confidence must be float in [0,1].")
                return GuardrailAudit(passed=False, violations=violations)
            if raw_json["priority_flag"] not in ("none", "emerg", "ban"):
                violations.append("SchemaError: priority_flag must be in {none,emerg,ban}.")
                return GuardrailAudit(passed=False, violations=violations)

        rationale_words = len(str(raw_json["rationale"]).split())
        if rationale_words > 40:
            violations.append(f"SchemaError: rationale exceeds 40 words ({rationale_words}).")

        link_ids = [int(a) for a in raw_json["link_ids"]]
        severity = int(raw_json["severity"])
        phi = float(raw_json["phi"])
        duration = int(raw_json["duration"])
        confidence = float(raw_json["confidence"])
        priority_flag = str(raw_json["priority_flag"])

        # Gate 1: Topological existence (a \in A_r \cup \partial A_r)
        invalid_links = [a for a in link_ids if a not in allowed_links]
        if invalid_links:
            violations.append(f"Gate1(Topology): hallucinated or out-of-zone link_ids {invalid_links}.")

        # Gate 4 check precedes Gate 2 floor relaxation when official road ban is active
        if priority_flag in ("emerg", "ban") and not dispatch.is_official_agency:
            violations.append(
                f"Gate4(Authorization): priority_flag='{priority_flag}' requires authenticated agency dispatch."
            )

        # Gate 2: Physical capacity and risk bounds
        if priority_flag == "ban" and dispatch.is_official_agency:
            if not (self.cfg.phi_ban_closure - 1e-6 <= phi <= self.cfg.phi_min_routine):
                violations.append(
                    f"Gate2(CapacityBound): official ban expects phi near {self.cfg.phi_ban_closure}, got {phi}."
                )
        else:
            if not (self.cfg.phi_min_routine <= phi <= 1.0):
                violations.append(
                    f"Gate2(CapacityBound): routine phi={phi:.3f} outside [{self.cfg.phi_min_routine}, 1.0]."
                )

        # Gate 3: Asymmetric temporal ramp bound
        for a in link_ids:
            if 0 <= a < len(prev_phi):
                delta_phi = phi - float(prev_phi[a])
                if delta_phi > self.cfg.ramp_recovery_cap + 1e-9:
                    violations.append(
                        f"Gate3(RampBound): link {a} upward jump +{delta_phi:.3f} exceeds {self.cfg.ramp_recovery_cap}."
                    )
                if (not is_incident_onset) and abs(delta_phi) > self.cfg.ramp_recovery_cap + 1e-9:
                    violations.append(
                        f"Gate3(RampBound): uncorroborated fluctuation |{delta_phi:.3f}| > {self.cfg.ramp_recovery_cap} on link {a}."
                    )

        if violations:
            return GuardrailAudit(passed=False, violations=violations)

        return GuardrailAudit(
            passed=True,
            payload=IncidentPayload(
                link_ids=link_ids,
                severity=severity,
                phi=phi,
                duration=duration,
                confidence=confidence,
                priority_flag=priority_flag,
                rationale=str(raw_json["rationale"]),
            ),
        )

    def map_payload_to_tensor(
        self,
        payload: IncidentPayload,
        tensor: PerturbationTensor,
    ) -> PerturbationTensor:
        """Deterministically project verified payload onto compact domain D_\\Xi (Table 3 & Eq. 3)."""
        r_base = SEVERITY_RISK_MAP[payload.severity] * self.cfg.r_max
        r_tilde = float(np.clip(payload.confidence * r_base, 0.0, self.cfg.r_max))

        for a in payload.link_ids:
            if payload.priority_flag == "ban":
                # Gate 4 verified road ban enforces \phi_ban = 0.05 directly, bypassing shrinkage
                phi_eff = self.cfg.phi_ban_closure
            else:
                phi_eff = 1.0 - payload.confidence * (1.0 - payload.phi)
                phi_eff = float(np.clip(phi_eff, self.cfg.phi_min_routine, 1.0))

            tensor.phi[a] = min(tensor.phi[a], phi_eff)
            tensor.hazard_risk[a] = max(tensor.hazard_risk[a], r_tilde)

            if payload.priority_flag == "emerg":
                # Set non-emergency priority corridor penalty \zeta_{a,m} for HDV/UE/FO/COOP
                tensor.zeta_bias[a, :] = np.clip(
                    tensor.zeta_bias[a, :] + 0.75 * self.cfg.zeta_max,
                    -self.cfg.zeta_max,
                    self.cfg.zeta_max,
                )
        return tensor

    def physics_prior_fallback(
        self,
        zone_links: Sequence[int],
        realized_speeds: np.ndarray,
        free_flow_speeds: np.ndarray,
        tensor: PerturbationTensor,
    ) -> PerturbationTensor:
        """Conservative fallback \\phi_a^{prior} = clip(v_a / v_a^{ff}, \\phi_min, 1.0)."""
        for a in zone_links:
            ratio = float(realized_speeds[a]) / max(float(free_flow_speeds[a]), 1e-6)
            phi_prior = float(np.clip(ratio, self.cfg.phi_min_routine, 1.0))
            tensor.phi[a] = min(tensor.phi[a], phi_prior)
            tensor.hazard_risk[a] = max(
                tensor.hazard_risk[a],
                float(np.clip((1.0 - phi_prior) * self.cfg.r_max, 0.0, self.cfg.r_max)),
            )
        return tensor
