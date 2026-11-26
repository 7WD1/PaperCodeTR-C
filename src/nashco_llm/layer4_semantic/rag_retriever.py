"""RAG historical incident retrieval over US-Accidents FAISS index (Section 3.2.1).

Uses OpenAI text-embedding-3-small embeddings into a FAISS inner-product index
(top-8 cases with cosine >= 0.78). Falls back cleanly to zero-shot parsing when
zero historical analogs pass the similarity threshold (Case Study C).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

import numpy as np


@dataclass
class HistoricalIncidentRecord:
    """Indexed historical US-Accidents record with severity-derived \\phi reference."""

    record_id: str
    text: str
    severity: int
    lane_blockage_ratio: float
    reference_phi: float
    embedding: Optional[np.ndarray] = None


class IncidentRAGRetriever:
    """FAISS inner-product retriever with cosine >= 0.78 filter and zero-shot fallback."""

    def __init__(
        self,
        records: Sequence[HistoricalIncidentRecord] = (),
        top_k: int = 8,
        cosine_threshold: float = 0.78,
        embed_fn: Optional[Callable[[str], np.ndarray]] = None,
    ) -> None:
        self.records = list(records)
        self.top_k = top_k
        self.cosine_threshold = cosine_threshold
        self.embed_fn = embed_fn or self._default_bag_of_chars_embedding
        self._matrix: Optional[np.ndarray] = None
        self._faiss_index = None
        if self.records:
            self._build_index()

    def _build_index(self) -> None:
        vecs = []
        for rec in self.records:
            v = rec.embedding if rec.embedding is not None else self.embed_fn(rec.text)
            norm = max(float(np.linalg.norm(v)), 1e-12)
            vecs.append(np.asarray(v, dtype=np.float32) / norm)
        mat = np.vstack(vecs).astype(np.float32)
        self._matrix = mat
        try:
            import faiss  # type: ignore

            index = faiss.IndexFlatIP(mat.shape[1])
            index.add(mat)
            self._faiss_index = index
        except ImportError:
            self._faiss_index = None

    def retrieve(self, query_text: str) -> List[tuple[HistoricalIncidentRecord, float]]:
        """Retrieve up to top_k analogs with cosine >= threshold; empty list -> zero-shot."""
        if not self.records or self._matrix is None:
            return []
        q = np.asarray(self.embed_fn(query_text), dtype=np.float32)
        q = q / max(float(np.linalg.norm(q)), 1e-12)
        k = min(self.top_k, len(self.records))

        if self._faiss_index is not None:
            scores, indices = self._faiss_index.search(q.reshape(1, -1), k)
            pairs = [(int(i), float(s)) for i, s in zip(indices[0], scores[0]) if i >= 0]
        else:
            sims = self._matrix @ q
            top_idx = np.argsort(sims)[::-1][:k]
            pairs = [(int(i), float(sims[i])) for i in top_idx]

        return [
            (self.records[idx], score)
            for idx, score in pairs
            if score >= self.cosine_threshold
        ]

    @staticmethod
    def map_severity_to_phi(severity: int, lane_blockage_ratio: float) -> float:
        """Map historical US-Accidents severity & lane blockage ratio to reference \\phi_a."""
        sev_base = {1: 0.90, 2: 0.78, 3: 0.60, 4: 0.35}.get(int(severity), 0.75)
        phi = 1.0 - max(0.0, min(0.8, 0.5 * (1.0 - sev_base) + 0.5 * lane_blockage_ratio))
        return float(np.clip(phi, 0.20, 1.0))

    @staticmethod
    def _default_bag_of_chars_embedding(text: str, dim: int = 64) -> np.ndarray:
        """Deterministic lightweight embedding for offline unit tests without API keys."""
        vec = np.zeros(dim, dtype=np.float32)
        tokens = text.lower().split()
        for tok in tokens:
            h = sum((idx + 1) * ord(ch) for idx, ch in enumerate(tok)) % dim
            vec[h] += 1.0
        return vec
