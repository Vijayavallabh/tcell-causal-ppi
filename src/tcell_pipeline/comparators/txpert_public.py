"""TxPert-public external comparator (feat-010) — PUBLIC-ONLY: STRING PPI topology, never the proprietary
TxPert graph/checkpoints. If ``valence-labs/TxPert`` is importable we record that (a real wrap would need its
public checkpoint); otherwise we reimplement its shape as a sparse graph-attention aggregator over STRING.

The predictor treats each held-out target as a query that attends over its STRING neighbours' training
responses: attention weight ∝ softmax(edge confidence / temperature), prediction = the attention-weighted
mean of neighbour responses. This is the "sparse graph transformer over STRING" reduced to a single
score-attention head — public, deterministic, adapted to our common output schema. A target with no covered
neighbour returns a zero shift.

ponytail: single deterministic score-attention head, STRING only (GO co-annotation edges are the documented
public extension); swap for a learned multi-head transformer over STRING+GO — or wrap the upstream public
checkpoint — if the comparator needs to be stronger. Registers as its own comparator family.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from tcell_pipeline import config
from tcell_pipeline.baselines.simple_baselines import BaseBaseline, _np
from tcell_pipeline.comparators.stable_shift import STRING, source_adjacency

try:  # public package presence is recorded in the compatibility report; predict stays the public reimpl
    import txpert as _txpert  # type: ignore  # noqa: F401

    _TXPERT_AVAILABLE = True
except Exception:
    _TXPERT_AVAILABLE = False


class TxPertPublicAdapter(BaseBaseline):
    """STRING score-attention aggregator. ``fit(genes, z)`` places each training target's mean Δz on its
    node; ``predict(genes)`` returns the softmax-over-edge-score attention-weighted mean of a query's covered
    neighbours. Same ``(genes, z) -> (delta_z, delta_x)`` contract as the other graph comparators."""

    LICENSE = "Apache-2.0 (valence-labs/TxPert public components); reimplementation of the public path"
    EXPOSURE_CLASS = "public-only (STRING topology; GO co-annotation extension deferred)"
    PUBLIC_ONLY = True
    CHECKPOINT: str | None = None
    family = "txpert_public"
    wrapped = _TXPERT_AVAILABLE

    def __init__(self, adjacency, gene_to_idx: dict[str, int], basis=None, temperature: float = 0.5) -> None:
        super().__init__(basis)
        self.gene_to_idx = gene_to_idx
        self.temperature = float(temperature)
        self._w = sp.csr_matrix(adjacency)
        self._n = self._w.shape[0]
        self._signal: np.ndarray | None = None    # (n, K) per-node mean training Δz
        self._presence: np.ndarray | None = None   # (n,) 1 where a node carries a training response

    @classmethod
    def from_hetero_graph(cls, graph=None, gene_to_idx: dict[str, int] | None = None, *, basis=None,
                          string_only: bool = True, **kw) -> "TxPertPublicAdapter":
        sources = (STRING,) if string_only else None
        a, g2i = source_adjacency(graph, gene_to_idx, sources=sources)
        return cls(a, g2i, basis=basis, **kw)

    def fit(self, genes, z, conditions=None) -> "TxPertPublicAdapter":
        z = _np(z)
        self._k = z.shape[1]
        s0 = np.zeros((self._n, self._k))
        counts = np.zeros(self._n)
        for g, row in zip(genes, z):
            j = self.gene_to_idx.get(g)
            if j is None:
                continue
            s0[j] += row
            counts[j] += 1.0
        seen = counts > 0
        s0[seen] /= counts[seen, None]
        self._signal = s0
        self._presence = seen
        return self

    def predict(self, genes, conditions=None) -> tuple[np.ndarray, np.ndarray]:
        if self._signal is None:
            raise RuntimeError("TxPertPublicAdapter.predict called before fit")
        dz = np.zeros((len(genes), self._k))
        for i, g in enumerate(genes):
            j = self.gene_to_idx.get(g)
            if j is None:
                continue
            row = self._w.getrow(j)
            nbrs, scores = row.indices, row.data
            covered = self._presence[nbrs]
            nbrs, scores = nbrs[covered], scores[covered]
            if nbrs.size == 0:
                continue
            attn = _softmax(scores / self.temperature)
            dz[i] = attn @ self._signal[nbrs]
        return dz, self._decode_genes(dz)


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()                      # stabilised softmax; a single neighbour -> weight 1
    e = np.exp(x)
    return e / e.sum()
