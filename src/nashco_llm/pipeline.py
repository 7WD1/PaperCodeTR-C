"""Algorithm 1: NashCo-LLM Online Rolling-Horizon Control and Clearing Loop.

Integrates all four architectural layers end-to-end:
- Layer 2: Disjunctive event trigger Eqs. (1)-(2) over K-zone MFD partition.
- Layer 4: Event-triggered RAG-LLM schema extraction + 4-gate physical guardrail Eq. (3)
           + periodic (every 3 steps) dual-agent behavioral calibration Eq. (11).
- Layer 1 & 3: Synchronous inner equilibrium loop (0 LLM calls) pairing MSC credit
               pricing Eqs. (13)-(14) with composite Mirror-Prox VI Eqs. (9)-(10).
- Layer 1-3: Stratified Owen-Shapley attribution Eqs. (16)-(18) + scale-invariant
             penalized \\varepsilon-core QP & budget-balanced clearing Eqs. (19)-(22).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from .config import NashCoConfig
from .layer1_coordinator import (
    MSCCreditPricer,
    ScaleInvariantCoreClearer,
    StratifiedOwenEstimator,
)
from .layer2_regional import (
    DisjunctiveEventTrigger,
    GazetteerGeofenceMatcher,
    MFDRegionalPartitioner,
)
from .layer3_equilibrium import (
    CompositeMirrorProxSolver,
    SemanticTrafficCostModel,
    VIResult,
)
from .layer3_equilibrium.mirror_prox_vi import ODClassBlock
from .layer4_semantic import (
    DualAgentCalibrator,
    FourGatePhysicalGuardrail,
    IncidentRAGRetriever,
    SemanticIncidentParser,
)
from .simulators.base import NetworkTopology, TrafficSimulator
from .types import ClearingResult, IncidentDispatch, PerturbationTensor, UserClass


@dataclass
class PeriodRecord:
    """Telemetry recorded at each rolling control step t in {1, ..., H}."""

    period: int
    triggered_zones: List[int]
    llm_calls_step: int
    eta_eff: float
    compliance: float
    tstt: float
    vi_iterations: int
    vi_residual: float
    clearing: ClearingResult
    sync_latency_s: float
    async_owen_latency_s: float


@dataclass
class EpisodeTrace:
    """Complete rolling-horizon rollout trace."""

    periods: List[PeriodRecord] = field(default_factory=list)
    link_flow_history: List[np.ndarray] = field(default_factory=list)

    @property
    def mean_tstt(self) -> float:
        return float(np.mean([p.tstt for p in self.periods])) if self.periods else 0.0

    @property
    def total_llm_calls(self) -> int:
        return int(sum(p.llm_calls_step for p in self.periods))


class NashCoLLMPipeline:
    """End-to-end four-layer NashCo-LLM controller executing Algorithm 1."""

    def __init__(
        self,
        topology: NetworkTopology,
        simulator: TrafficSimulator,
        blocks: Sequence[ODClassBlock],
        alpha_vot: np.ndarray,
        config: NashCoConfig,
        keyword_gazetteer: Optional[Dict[str, List[int]]] = None,
        rag_retriever: Optional[IncidentRAGRetriever] = None,
        use_full_tier1_eta: bool = False,
    ) -> None:
        self.topology = topology
        self.simulator = simulator
        self.blocks = list(blocks)
        self.alpha_vot = np.asarray(alpha_vot, dtype=float)
        self.cfg = config
        self.use_full_tier1_eta = use_full_tier1_eta

        # Layer 2: Regional partition & disjunctive event trigger
        self.partitioner = MFDRegionalPartitioner(config.regional.num_zones_K)
        self.partition = self.partitioner.partition_links(
            num_links=len(topology.capacities),
            link_endpoints=topology.link_endpoints,
            num_members=len(self.blocks),
        )
        self.matcher = GazetteerGeofenceMatcher(
            keyword_to_links=keyword_gazetteer or {},
            link_to_zone=self.partition.link_to_zone,
        )
        self.trigger = DisjunctiveEventTrigger(config.regional)

        # Layer 3: Semantic cost model & composite Mirror-Prox VI solver
        self.cost_model = SemanticTrafficCostModel(
            topology.free_flow_times, topology.capacities, config.equilibrium
        )
        self.vi_solver = CompositeMirrorProxSolver(
            cost_model=self.cost_model,
            path_link_incidence=topology.path_link_incidence,
            blocks=self.blocks,
            alpha_vot=self.alpha_vot,
            config=config.equilibrium,
        )

        # Layer 4: Semantic incident parser, 4-gate guardrail & dual-agent calibrator
        self.retriever = rag_retriever or IncidentRAGRetriever()
        self.guardrail = FourGatePhysicalGuardrail(config.semantic)
        self.parser = SemanticIncidentParser(config.semantic, self.retriever, self.guardrail)
        self.calibrator = DualAgentCalibrator(config.semantic)

        # Layer 1: MSC credit pricer, stratified Owen estimator & scale-invariant QP clearer
        self.pricer = MSCCreditPricer(config.cooperative)
        self.owen_estimator = StratifiedOwenEstimator(
            coalition_blocks=self.partition.coalition_blocks,
            sample_budget_m=config.cooperative.owen_samples_M,
            seed=config.seed,
        )
        self.clearer = ScaleInvariantCoreClearer(config.cooperative)

        # Determine which user classes represent atomic commercial fleets
        self.atomic_class_flags = [False] * len(self.alpha_vot)
        for b in self.blocks:
            if b.is_atomic_fleet or b.user_class == UserClass.FO:
                self.atomic_class_flags[b.class_idx] = True

    def run_horizon(
        self,
        dispatches_by_period: Optional[Dict[int, List[IncidentDispatch]]] = None,
        base_compliance: float = 1.0,
        enable_msc: bool = True,
        enable_llm: bool = True,
    ) -> EpisodeTrace:
        """Execute Algorithm 1 across rolling periods t = 1, ..., H."""
        dispatches_map = dispatches_by_period or {}
        trace = EpisodeTrace()
        num_links = len(self.topology.capacities)
        num_classes = len(self.alpha_vot)

        xi = PerturbationTensor.nominal(num_links, num_classes)
        prev_phi = xi.phi.copy()
        warm_shares = self.vi_solver.uniform_initial_shares()
        _, init_flows, init_class_flows = self.vi_solver.shares_to_flows(warm_shares)
        link_state = self.simulator.load_flows(init_flows, init_class_flows, xi)
        compliance = float(base_compliance)

        for t in range(1, self.cfg.horizon.num_periods + 1):
            t_sync_start = time.perf_counter()
            calls_before = self.parser.stats.llm_calls

            # --- Layer 2: Disjunctive Event Detection (Eqs. 1-2) ---
            step_dispatches = dispatches_map.get(t, [])
            decisions = self.trigger.evaluate_all_zones(
                link_state=link_state,
                zone_links=self.partition.zone_links,
                dispatches=step_dispatches,
                matcher=self.matcher,
            )
            triggered = [r for r, dec in decisions.items() if dec.activated]

            # --- Layer 4: Parallel Event-Triggered Semantic Outer Shell (Eq. 3) ---
            xi = PerturbationTensor.nominal(num_links, num_classes)
            if enable_llm:
                for r in triggered:
                    dec = decisions[r]
                    allowed = set(self.partition.zone_links[r]) | set(self.partition.zone_boundary_links[r])
                    if dec.matched_dispatches:
                        for disp in dec.matched_dispatches:
                            grounded_links, _ = self.matcher.ground_dispatch(disp)
                            xi, _ = self.parser.parse_and_verify(
                                zone_id=r,
                                allowed_links=allowed,
                                grounded_links=grounded_links,
                                dispatch=disp,
                                prev_phi=prev_phi,
                                realized_speeds=link_state.speeds_kmh,
                                free_flow_speeds=link_state.free_flow_speeds_kmh,
                                tensor=xi,
                            )
                    else:
                        xi = self.guardrail.physics_prior_fallback(
                            self.partition.zone_links[r],
                            link_state.speeds_kmh,
                            link_state.free_flow_speeds_kmh,
                            xi,
                        )
            prev_phi = xi.phi.copy()

            # --- Layer 1 & 3: Synchronous Inner Equilibrium Loop (0 LLM calls) ---
            eta_eff = (
                self.pricer.compute_adaptive_eta(compliance, use_full_tier1=self.use_full_tier1_eta)
                if enable_msc
                else 0.0
            )
            # Non-cooperative counterfactual for Stage 2 v(S)
            noncoop_res = self.vi_solver.solve(xi=xi, initial_shares=warm_shares, msc_rates_fn=None)

            def msc_fn(lf: np.ndarray, clf: np.ndarray, dc_dx: np.ndarray) -> np.ndarray:
                return self.pricer.compute_link_msc_rates(
                    link_flows=lf,
                    class_link_flows=clf,
                    dc_dx=dc_dx,
                    alpha_vot=self.alpha_vot,
                    atomic_class_flags=self.atomic_class_flags,
                    eta_eff=eta_eff,
                )

            coop_res: VIResult = self.vi_solver.solve(
                xi=xi,
                initial_shares=warm_shares,
                msc_rates_fn=msc_fn if enable_msc else None,
            )
            warm_shares = list(coop_res.path_shares)
            link_state = self.simulator.load_flows(coop_res.link_flows, coop_res.class_link_flows, xi)
            t_sync_end = time.perf_counter()

            # --- Layer 1-3: Cooperative Clearing (Asynchronous Owen + Synchronous QP) ---
            t_owen_start = time.perf_counter()
            clearing = self._run_cooperative_clearing(noncoop_res, coop_res, xi, eta_eff)
            t_owen_end = time.perf_counter()

            # --- Layer 4: Periodic Dual-Agent Behavioral Calibration (Eq. 11) ---
            if self.calibrator.should_calibrate(t):
                self.alpha_vot, xi.zeta_bias = self.calibrator.update(
                    self.alpha_vot, xi.zeta_bias, observed_compliance=compliance
                )
                self.vi_solver.alpha_vot = self.alpha_vot

            compliance = self.pricer.update_compliance_retention(base_compliance, clearing.core_violation)
            tstt = float(np.sum(link_state.flows * link_state.travel_times))

            trace.periods.append(
                PeriodRecord(
                    period=t,
                    triggered_zones=triggered,
                    llm_calls_step=self.parser.stats.llm_calls - calls_before,
                    eta_eff=eta_eff,
                    compliance=compliance,
                    tstt=tstt,
                    vi_iterations=coop_res.iterations,
                    vi_residual=coop_res.final_residual,
                    clearing=clearing,
                    sync_latency_s=t_sync_end - t_sync_start,
                    async_owen_latency_s=t_owen_end - t_owen_start,
                )
            )
            trace.link_flow_history.append(coop_res.link_flows.copy())

        return trace

    def _run_cooperative_clearing(
        self,
        noncoop_res: VIResult,
        coop_res: VIResult,
        xi: PerturbationTensor,
        eta_eff: float,
    ) -> ClearingResult:
        """Execute Stage 2 surrogate Owen-Shapley + penalized \\varepsilon-core QP."""
        num_members = len(self.blocks)
        num_links = len(self.topology.capacities)

        x_nc_mem = np.zeros((num_members, num_links), dtype=float)
        x_co_mem = np.zeros((num_members, num_links), dtype=float)
        for idx, b in enumerate(self.blocks):
            p_nc = noncoop_res.path_shares[idx]
            p_co = coop_res.path_shares[idx]
            for local_i, pid in enumerate(b.path_indices):
                x_nc_mem[idx] += b.demand * float(p_nc[local_i]) * self.topology.path_link_incidence[pid]
                x_co_mem[idx] += b.demand * float(p_co[local_i]) * self.topology.path_link_incidence[pid]

        c_nc = self.cost_model.link_augmented_cost(noncoop_res.link_flows, xi)
        c_co = self.cost_model.link_augmented_cost(coop_res.link_flows, xi)
        dc_dx_nc = self.cost_model.link_cost_derivative(noncoop_res.link_flows, xi)
        dc_dx_co = self.cost_model.link_cost_derivative(coop_res.link_flows, xi)

        member_vot = np.array([self.alpha_vot[b.class_idx] for b in self.blocks], dtype=float)
        noncoop_costs = member_vot * (x_nc_mem @ c_nc)
        coop_costs = member_vot * (x_co_mem @ c_co)
        grand_surplus = float(np.sum(noncoop_costs - coop_costs))

        pi_mat = self.pricer.compute_link_msc_rates(
            coop_res.link_flows, coop_res.class_link_flows, dc_dx_co,
            self.alpha_vot, self.atomic_class_flags, eta_eff,
        )
        collected_charges = np.array(
            [float(np.dot(x_co_mem[idx], pi_mat[:, b.class_idx])) for idx, b in enumerate(self.blocks)],
            dtype=float,
        )

        v_hat = self.owen_estimator.build_sensitivity_surrogate(
            x_noncoop_by_member=x_nc_mem,
            delta_x_by_member=x_co_mem - x_nc_mem,
            link_cost_fn=lambda x: self.cost_model.link_augmented_cost(x, xi),
            dc_dx_noncoop=dc_dx_nc,
        )
        owen_vals = self.owen_estimator.estimate(v_hat)
        cuts = self.clearer.screen_critical_coalitions(
            num_members=num_members,
            coalition_blocks=self.partition.coalition_blocks,
            owen_values=owen_vals,
            v_oracle=v_hat,
        )
        return self.clearer.clear_epoch(
            owen_values=owen_vals,
            grand_surplus=grand_surplus,
            critical_cuts=cuts,
            noncoop_costs=noncoop_costs,
            coop_costs=coop_costs,
            collected_msc_charges=collected_charges,
        )
