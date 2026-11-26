"""MFD-weighted regional graph partitioning (Paper Section 3.1 & Section 3.3.2).

Divides directed road graph G=(V, A) into K contiguous spatial zones B = {B_1, ..., B_K}
and anchors cross-regional commercial fleets q to their dominant bottleneck zone:
    r*(q) = argmax_r sum_{a in A_r} x_{a,q}^{Coop}.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class RegionalPartition:
    """Contiguous K-zone spatial decomposition of links and coalition units."""

    num_zones: int
    zone_links: Dict[int, List[int]]          # r -> A_r
    zone_boundary_links: Dict[int, List[int]] # r -> \partial A_r (1-hop boundary)
    link_to_zone: Dict[int, int]
    coalition_blocks: List[List[int]]         # B = {B_1, ..., B_K} over M


class MFDRegionalPartitioner:
    """Multilevel METIS / spectral MFD-weighted regional graph partitioner."""

    def __init__(self, num_zones_k: int = 8) -> None:
        if num_zones_k < 1:
            raise ValueError("num_zones_k must be >= 1.")
        self.num_zones_k = num_zones_k

    def partition_links(
        self,
        num_links: int,
        link_endpoints: Sequence[Tuple[int, int]],
        mfd_similarity_weights: Optional[np.ndarray] = None,
        member_zone_hints: Optional[Sequence[int]] = None,
        num_members: Optional[int] = None,
    ) -> RegionalPartition:
        """Partition road links into K contiguous zones using pymetis or spectral fallback."""
        k = min(self.num_zones_k, max(1, num_links))
        adjacency = self._build_line_graph_adjacency(num_links, link_endpoints)
        link_assignments = self._solve_partition(num_links, adjacency, k, mfd_similarity_weights)

        zone_links: Dict[int, List[int]] = {r: [] for r in range(k)}
        link_to_zone: Dict[int, int] = {}
        for link_id, zone_id in enumerate(link_assignments):
            z = int(zone_id)
            zone_links[z].append(link_id)
            link_to_zone[link_id] = z

        # One-hop boundary links \partial A_r for Gate 1 topological validation
        zone_boundary: Dict[int, List[int]] = {r: [] for r in range(k)}
        for r in range(k):
            boundary_set = set()
            for a in zone_links[r]:
                for nb in adjacency[a]:
                    if link_to_zone[nb] != r:
                        boundary_set.add(nb)
            zone_boundary[r] = sorted(boundary_set)

        n_m = num_members if num_members is not None else max(k, num_links)
        blocks: List[List[int]] = [[] for _ in range(k)]
        for member_id in range(n_m):
            if member_zone_hints is not None and member_id < len(member_zone_hints):
                z = int(member_zone_hints[member_id]) % k
            else:
                z = member_id % k
            blocks[z].append(member_id)

        return RegionalPartition(
            num_zones=k,
            zone_links=zone_links,
            zone_boundary_links=zone_boundary,
            link_to_zone=link_to_zone,
            coalition_blocks=blocks,
        )

    @staticmethod
    def anchor_fleet_to_zone(
        fleet_link_flows: np.ndarray,
        zone_links: Dict[int, List[int]],
    ) -> int:
        """Anchor cross-regional commercial fleet q to dominant bottleneck zone r*(q)."""
        best_zone = 0
        best_flow = -1.0
        for zone_id, links in zone_links.items():
            zone_total = float(np.sum(fleet_link_flows[links])) if links else 0.0
            if zone_total > best_flow:
                best_flow = zone_total
                best_zone = zone_id
        return best_zone

    @staticmethod
    def _build_line_graph_adjacency(
        num_links: int,
        link_endpoints: Sequence[Tuple[int, int]],
    ) -> List[List[int]]:
        node_to_links: Dict[int, List[int]] = {}
        for link_id, (u, v) in enumerate(link_endpoints):
            node_to_links.setdefault(u, []).append(link_id)
            node_to_links.setdefault(v, []).append(link_id)
        adj: List[List[int]] = [[] for _ in range(num_links)]
        for link_id, (u, v) in enumerate(link_endpoints):
            neighbors = set(node_to_links.get(u, [])) | set(node_to_links.get(v, []))
            neighbors.discard(link_id)
            adj[link_id] = sorted(neighbors)
        return adj

    @staticmethod
    def _solve_partition(
        num_links: int,
        adjacency: List[List[int]],
        k: int,
        weights: Optional[np.ndarray],
    ) -> np.ndarray:
        if k == 1:
            return np.zeros(num_links, dtype=int)
        try:
            import pymetis  # type: ignore

            _, membership = pymetis.part_graph(k, adjacency=adjacency)
            return np.asarray(membership, dtype=int)
        except ImportError:
            # Deterministic Laplacian Fiedler / quantile fallback when pymetis is absent
            lap = np.zeros((num_links, num_links), dtype=float)
            for i, nbs in enumerate(adjacency):
                for j in nbs:
                    w = 1.0 if weights is None else float(weights[i, j])
                    lap[i, j] -= w
                    lap[i, i] += w
            eigvals, eigvecs = np.linalg.eigh(lap)
            fiedler = eigvecs[:, min(1, num_links - 1)]
            order = np.argsort(fiedler)
            membership = np.zeros(num_links, dtype=int)
            for rank, idx in enumerate(order):
                membership[idx] = min(k - 1, (rank * k) // num_links)
            return membership
