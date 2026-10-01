"""Stable-Shift external comparator (feat-010), REIMPLEMENTED — the published code is unconfirmed, so
this is a clean-room predictor matching the method's described shape, adapted to our data format, splits,
and common output schema (report §Comparators: "identical target IDs, fold-local basis, split hashes").

The idea: learn a fold-local LOW-RANK program subspace from the TRAIN responses only (the "stable" part —
a truncated-SVD basis that cannot chase directions unsupported by training), place each training target's
reduced coordinate on its protein node, then predict a held-out target's shift by one graph convolution over
the STRING topology (a presence-weighted mean of its neighbours' reduced coordinates) decoded back through
the low-rank basis. A target with no covered neighbour returns a zero shift. Topology + a train-only basis;
no proprietary data, no per-gene supervision beyond the training responses.

ponytail: single graph-conv hop + a fixed-rank SVD; raise the hop count or rank, or swap the one-step conv
for a learned propagation, if the comparator needs to be stronger. STRING-only by default (the public PPI
channel); ``string_only=False`` unions all PPI sources when a denser graph helps.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from tcell_pipeline import config
from tcell_pipeline.baselines.simple_baselines import BaseBaseline, _np
from tcell_pipeline.graph import PROTEIN, build_hetero_graph

_PP_RELATIONS = ("physical_ppi", "co_complex", "functional_assoc")
_SCORE_COL = len(config.PPI_SOURCES)                 # edge_attr layout: source one-hot(5) then score at idx 5
_SOURCE_INDEX = {s: i for i, s in enumerate(config.PPI_SOURCES)}  # bioplex/huri/biogrid/string/corum -> 0..4
STRING = "string"


def source_adjacency(graph, gene_to_idx=None, *, sources: tuple[str, ...] | None = ("string",)
                     ) -> tuple[sp.csr_matrix, dict]:
    """Symmetric weighted PPI adjacency restricted to the given evidence ``sources`` (None = all sources).

    Unions the three protein-protein relations, keeps only edges whose source one-hot has a bit set for one
    of ``sources``, weights each by its confidence score, and symmetrises. ``sources=("string",)`` gives the
    public STRING channel the comparators run on. Any None loads the real graph from config paths."""
    if graph is None:
        graph, gene_to_idx = build_hetero_graph()
    gene_to_idx = gene_to_idx if gene_to_idx is not None else graph.gene_to_idx
    cols = None if sources is None else [_SOURCE_INDEX[s] for s in sources]
    n = graph[PROTEIN].x.shape[0]
    rows, colidx, wts = [], [], []
    for rel in _PP_RELATIONS:
        ei = graph[PROTEIN, rel, PROTEIN].edge_index
        ea = graph[PROTEIN, rel, PROTEIN].edge_attr
        if ei.numel() == 0:
            continue
        score = ea[:, _SCORE_COL].numpy()
        if cols is None:
            keep = np.ones(ei.shape[1], dtype=bool)
        else:
            keep = (ea[:, cols].numpy() > 0).any(1)
        if not keep.any():
            continue
        rows.append(ei[0].numpy()[keep])
        colidx.append(ei[1].numpy()[keep])
        wts.append(score[keep])
    if rows:
        r, c, w = np.concatenate(rows), np.concatenate(colidx), np.concatenate(wts)
    else:
        r = c = w = np.zeros(0)
    a = sp.coo_matrix((w, (r, c)), shape=(n, n)).tocsr()
    return (a + a.T).tocsr(), gene_to_idx


class StableShiftAdapter(BaseBaseline):
    """Fold-local low-rank + STRING graph-conv comparator. ``fit(genes, z)`` learns the train-only SVD
    subspace and the per-node reduced coordinates; ``predict(genes)`` graph-convolves them one hop and
    decodes. Same ``(genes, z) -> (delta_z, delta_x)`` contract as ``NetworkPropagationBaseline`` so it
    drops into the screening / sealed scorers unchanged."""

    LICENSE = "reimplementation; original Stable-Shift license unconfirmed"
    EXPOSURE_CLASS = "public-reimplementation"
    PUBLIC_ONLY = True
    CHECKPOINT: str | None = None
    family = "stable_shift"
    wrapped = False

    def __init__(self, adjacency, gene_to_idx: dict[str, int], basis=None, rank: int = 8) -> None:
        super().__init__(basis)
        self.gene_to_idx = gene_to_idx
        self.rank = int(rank)
        a = sp.csr_matrix(adjacency)
        self._w = a + sp.eye(a.shape[0], format="csr")   # self-loop: a training node keeps its own coord
        self._n = self._w.shape[0]
        self._components: np.ndarray | None = None       # (r, K) train-only SVD directions
        self._mean: np.ndarray | None = None             # (K,) train response mean
        self._reduced: np.ndarray | None = None          # (n, r) per-node reduced coords (0 where absent)
        self._presence: np.ndarray | None = None         # (n,) 1 where a node carries a training response

    @classmethod
    def from_hetero_graph(cls, graph=None, gene_to_idx: dict[str, int] | None = None, *, basis=None,
                          string_only: bool = True, **kw) -> "StableShiftAdapter":
        sources = (STRING,) if string_only else None
        a, g2i = source_adjacency(graph, gene_to_idx, sources=sources)
        return cls(a, g2i, basis=basis, **kw)

    def fit(self, genes, z, conditions=None) -> "StableShiftAdapter":
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
        s0[seen] /= counts[seen, None]                   # mean Δz per training-target node
        self._presence = seen.astype(np.float64)
        signal = s0[seen]
        if signal.shape[0] == 0:                         # no training target hit the graph -> zero predictor
            self._mean = np.zeros(self._k)
            self._components = np.zeros((0, self._k))
            self._reduced = np.zeros((self._n, 0))
            return self
        self._mean = signal.mean(0)
        r = min(self.rank, signal.shape[0], signal.shape[1])
        _, _, vt = np.linalg.svd(signal - self._mean, full_matrices=False)
        self._components = vt[:r]                         # (r, K)
        self._reduced = np.zeros((self._n, r))
        self._reduced[seen] = (s0[seen] - self._mean) @ self._components.T
        return self

    def predict(self, genes, conditions=None) -> tuple[np.ndarray, np.ndarray]:
        if self._components is None:
            raise RuntimeError("StableShiftAdapter.predict called before fit")
        # one graph-conv hop: presence-weighted neighbour mean of the reduced coords (self-loop included)
        num = self._w @ self._reduced                    # (n, r)
        den = np.asarray(self._w @ self._presence).reshape(-1)  # (n,) covered mass reaching each node
        with np.errstate(invalid="ignore", divide="ignore"):
            smoothed = np.divide(num, den[:, None], out=np.zeros_like(num), where=den[:, None] > 0)
        dz = np.zeros((len(genes), self._k))
        for i, g in enumerate(genes):
            j = self.gene_to_idx.get(g)
            if j is not None and den[j] > 1e-12 and self._components.shape[0]:
                dz[i] = smoothed[j] @ self._components + self._mean
        return dz, self._decode_genes(dz)
