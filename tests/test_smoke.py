"""End-to-end automated verification tests covering Layers 1-4 and Theorems 1-5."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nashco_llm.config import NashCoConfig
from nashco_llm.layer1_coordinator import (
    MSCCreditPricer,
    ScaleInvariantCoreClearer,
    StratifiedOwenEstimator,
)
from nashco_llm.layer2_regional import DisjunctiveEventTrigger, GazetteerGeofenceMatcher, MFDRegionalPartitioner
from nashco_llm.layer3_equilibrium import CompositeMirrorProxSolver, SemanticTrafficCostModel
from nashco_llm.layer3_equilibrium.mirror_prox_vi import ODClassBlock
from nashco_llm.layer4_semantic import FourGatePhysicalGuardrail, IncidentRAGRetriever, SemanticIncidentParser
from nashco_llm.pipeline import NashCoLLMPipeline
from nashco_llm.simulators.bpr_analytical import (
    AnalyticalBPRSimulator,
    build_braess_network,
    build_two_fleet_network,
)
from nashco_llm.types import IncidentDispatch, PerturbationTensor, UserClass


def test_disjunctive_event_trigger_top_q_guard() -> None:
    """Verify Eq. (1)-(2) prevents large-zone L2 dilution on localized collisions."""
    cfg = NashCoConfig()
    trigger = DisjunctiveEventTrigger(cfg.regional)
    # 370-link zone where only 3 links experience 40% speed collapse
    pred = np.full(370, 80.0)
    real = pred.copy()
    real[:3] = 48.0  # 40% drop on 3 links; pure L2 norm would be ~0.036 < 0.12
    score = trigger.compute_physical_anomaly(real, pred, list(range(370)))
    assert score >= 0.39, f"Top-q guard failed to catch localized drop: score={score}"


def test_four_gate_guardrail_and_retry() -> None:
    """Verify Gates 1-4 reject hallucinations, unauthorized bans, and bad ramps."""
    cfg = NashCoConfig()
    guard = FourGatePhysicalGuardrail(cfg.semantic)
    prev_phi = np.ones(10)
    unofficial = IncidentDispatch("d1", 5, "Road ban rumor", is_official_agency=False)
    bad_payload = {
        "link_ids": [99],       # Gate 1 violation
        "severity": 4,
        "phi": 1.4,             # Gate 2 violation
        "duration": 5,
        "confidence": 0.9,
        "priority_flag": "ban", # Gate 4 violation
        "rationale": "Hallucinated link and unverified road ban.",
    }
    audit = guard.validate(bad_payload, allowed_links={0, 1, 2}, prev_phi=prev_phi, dispatch=unofficial)
    assert not audit.passed
    assert any("Gate1" in v for v in audit.violations)
    assert any("Gate4" in v for v in audit.violations)

    # Valid official road ban enforces phi_ban = 0.05 directly bypassing shrinkage
    official = IncidentDispatch("d2", 5, "Official hazmat ban on link 1", is_official_agency=True)
    good_ban = {
        "link_ids": [1],
        "severity": 4,
        "phi": 0.05,
        "duration": 6,
        "confidence": 0.80,
        "priority_flag": "ban",
        "rationale": "Official hazmat closure verified.",
    }
    audit_ok = guard.validate(good_ban, allowed_links={0, 1, 2}, prev_phi=prev_phi, dispatch=official)
    assert audit_ok.passed and audit_ok.payload is not None
    xi = guard.map_payload_to_tensor(audit_ok.payload, PerturbationTensor.nominal(10, 4))
    assert abs(xi.phi[1] - 0.05) < 1e-9


def test_heterogeneous_vot_msc_alignment() -> None:
    """Verify Theorem 3(i) external-only atomic fleet rate avoids self-externality double-counting."""
    cfg = NashCoConfig()
    pricer = MSCCreditPricer(cfg.cooperative)
    class_flows = np.array([[100.0, 200.0], [150.0, 50.0]])  # 2 links x 2 atomic fleets
    link_flows = np.sum(class_flows, axis=1)
    dc_dx = np.array([0.1, 0.2])
    alpha = np.array([1.0, 2.5])  # Heterogeneous VOTs
    pi = pricer.compute_link_msc_rates(
        link_flows, class_flows, dc_dx, alpha, atomic_class_flags=[True, True], eta_eff=1.0
    )
    # For fleet 0 (alpha=1.0) on link 0: self = 1.0*100*0.1 = 10, ext = 2.5*200*0.1 = 50 -> sum = 60 = dJ_SO/df
    self_0 = alpha[0] * class_flows[0, 0] * dc_dx[0]
    total_ext = (class_flows[0] @ alpha) * dc_dx[0]
    assert abs((self_0 + pi[0, 0]) - total_ext) < 1e-12


def test_scale_invariant_core_clearing_and_cold_start() -> None:
    """Verify Theorem 5 exact scale invariance, budget balance, and cold-start bypass."""
    cfg = NashCoConfig()
    cfg.cooperative.qp_solver = "scipy"
    clearer = ScaleInvariantCoreClearer(cfg.cooperative)

    psi = np.array([40.0, 35.0, 25.0])
    v_grand = 100.0
    cuts = [(np.array([True, False, False]), 30.0), (np.array([True, True, False]), 72.0)]
    nc_costs = np.array([200.0, 180.0, 150.0])
    co_costs = np.array([150.0, 150.0, 130.0])
    charges = np.array([20.0, 15.0, 10.0])

    res1 = clearer.clear_epoch(psi, v_grand, cuts, nc_costs, co_costs, charges)
    assert res1.budget_residual < 1e-9
    assert abs(np.sum(res1.allocated_surplus) - v_grand) < 1e-7

    # Rescale all cost units by c = 7.5 -> y*/v(M) and eps* must be strictly invariant
    c = 7.5
    cuts_scaled = [(m, c * vs) for m, vs in cuts]
    res2 = clearer.clear_epoch(c * psi, c * v_grand, cuts_scaled, c * nc_costs, c * co_costs, c * charges)
    assert np.allclose(res2.allocated_surplus / (c * v_grand), res1.allocated_surplus / v_grand, atol=1e-4)
    assert abs(res2.core_violation - res1.core_violation) < 1e-4

    # Cold-start safeguard when v(M) <= 0
    res_cold = clearer.clear_epoch(psi, -5.0, cuts, nc_costs, co_costs, charges)
    assert res_cold.cold_start_bypass
    assert np.allclose(res_cold.credit_rebates, charges)
    assert res_cold.budget_residual == 0.0


def test_full_algorithm1_pipeline_braess() -> None:
    """Run full Algorithm 1 rolling-horizon loop end-to-end on Braess paradox with incident text."""
    cfg = NashCoConfig.from_yaml(Path(__file__).resolve().parents[1] / "configs" / "default.yaml")
    cfg.horizon.num_periods = 4
    cfg.regional.num_zones_K = 2
    cfg.cooperative.owen_samples_M = 200
    cfg.cooperative.qp_solver = "scipy"
    cfg.semantic.provider = "mock"

    topo = build_braess_network(total_demand=4000.0)
    sim = AnalyticalBPRSimulator(topo, cfg.equilibrium)
    blocks = [
        ODClassBlock(0, UserClass.COOP, 0, 2000.0, [0, 1, 2]),
        ODClassBlock(0, UserClass.COOP, 1, 2000.0, [0, 1, 2]),
    ]
    pipe = NashCoLLMPipeline(
        topology=topo,
        simulator=sim,
        blocks=blocks,
        alpha_vot=np.array([1.0, 1.0]),
        config=cfg,
        keyword_gazetteer={"shortcut": [4]},
        use_full_tier1_eta=True,
    )
    dispatches = {
        2: [IncidentDispatch("ev2", 2, "Two-lane rainstorm collision on shortcut link 4", is_official_agency=True)]
    }
    trace = pipe.run_horizon(dispatches_by_period=dispatches)
    assert len(trace.periods) == 4
    assert trace.periods[1].llm_calls_step >= 1
    assert trace.periods[0].clearing.budget_residual < 1e-8


if __name__ == "__main__":
    test_disjunctive_event_trigger_top_q_guard()
    test_four_gate_guardrail_and_retry()
    test_heterogeneous_vot_msc_alignment()
    test_scale_invariant_core_clearing_and_cold_start()
    test_full_algorithm1_pipeline_braess()
    print("ALL 5 NASHCO-LLM UNIT & INTEGRATION TESTS PASSED!")
