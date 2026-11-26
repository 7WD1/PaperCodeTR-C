"""Disjunctive event-triggered activation protocol (Paper Section 3.1, Eqs. 1-2)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Sequence, Set

import numpy as np

from ..config import RegionalConfig
from ..types import IncidentDispatch, LinkState


@dataclass
class TriggerDecision:
    """Zone-level activation output for rolling period t."""

    zone_id: int
    phys_anomaly_score: float    # \Delta_r^{phys}(t) in Eq. (1)
    has_text_event: bool         # NewTextEvent_r(t) in {0, 1}
    activated: bool              # I_r^{act}(t) in Eq. (2)
    matched_dispatches: List[IncidentDispatch]


class GazetteerGeofenceMatcher:
    """Sub-millisecond deterministic gazetteer and milepost-to-zone geofence matcher.

    Resolves highway mileposts and arterial cross-streets to TNTP link IDs and hosting
    zones prior to any LLM invocation (Section 3.1 & Section 3.2.1).
    """

    def __init__(
        self,
        keyword_to_links: Dict[str, List[int]],
        link_to_zone: Dict[int, int],
    ) -> None:
        self.keyword_to_links = {k.lower(): v for k, v in keyword_to_links.items()}
        self.link_to_zone = link_to_zone

    def ground_dispatch(self, dispatch: IncidentDispatch) -> tuple[List[int], Set[int]]:
        """Return (matched_link_ids, matched_zone_ids) without calling any LLM."""
        text_lower = dispatch.raw_text.lower()
        matched_links: Set[int] = set()
        matched_zones: Set[int] = set(dispatch.geofence_zones)

        for phrase, links in self.keyword_to_links.items():
            if phrase in text_lower:
                matched_links.update(links)

        # Parse explicit link references such as "link 16" or "corridor 16-18"
        for m in re.findall(r"\blink[s]?\s*(\d+)", text_lower):
            lid = int(m)
            if lid in self.link_to_zone:
                matched_links.add(lid)

        for lid in matched_links:
            if lid in self.link_to_zone:
                matched_zones.add(self.link_to_zone[lid])

        return sorted(matched_links), matched_zones


class DisjunctiveEventTrigger:
    """Evaluates Eqs. (1)-(2) across all K zones to suppress idle LLM inference."""

    def __init__(self, config: RegionalConfig) -> None:
        self.cfg = config

    def compute_physical_anomaly(
        self,
        realized_speeds: np.ndarray,
        predicted_speeds: np.ndarray,
        zone_link_ids: Sequence[int],
    ) -> float:
        """Compute Eq. (1): max of regional relative L2 drop and top-q localized drop."""
        if not zone_link_ids:
            return 0.0
        idx = np.asarray(zone_link_ids, dtype=int)
        v_r = realized_speeds[idx]
        v_hat = predicted_speeds[idx]

        denom_l2 = max(float(np.linalg.norm(v_hat)), self.cfg.v_min_kmh)
        l2_score = float(np.linalg.norm(v_r - v_hat)) / denom_l2

        rel_drops = np.abs(v_hat - v_r) / np.maximum(v_hat, self.cfg.v_min_kmh)
        q = min(self.cfg.top_q_links, len(rel_drops))
        top_q_score = float(np.mean(np.sort(rel_drops)[-q:]))
        return max(l2_score, top_q_score)

    def evaluate_all_zones(
        self,
        link_state: LinkState,
        zone_links: Dict[int, List[int]],
        dispatches: Sequence[IncidentDispatch],
        matcher: GazetteerGeofenceMatcher,
    ) -> Dict[int, TriggerDecision]:
        """Evaluate Eq. (2): I_r^{act}(t) = I(\\Delta_r^{phys} > \\vartheta_trig) OR NewTextEvent_r."""
        zone_dispatches: Dict[int, List[IncidentDispatch]] = {r: [] for r in zone_links}
        for disp in dispatches:
            _, zones = matcher.ground_dispatch(disp)
            for z in zones:
                if z in zone_dispatches:
                    zone_dispatches[z].append(disp)

        decisions: Dict[int, TriggerDecision] = {}
        for r, links in zone_links.items():
            score = self.compute_physical_anomaly(
                link_state.speeds_kmh,
                link_state.predicted_speeds_kmh,
                links,
            )
            has_text = len(zone_dispatches[r]) > 0
            activated = (score > self.cfg.trigger_threshold) or has_text
            decisions[r] = TriggerDecision(
                zone_id=r,
                phys_anomaly_score=score,
                has_text_event=has_text,
                activated=activated,
                matched_dispatches=zone_dispatches[r],
            )
        return decisions
