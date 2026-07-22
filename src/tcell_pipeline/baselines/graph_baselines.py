"""Graph baselines (feat-007): the three PPI-graph references the report requires alongside the simple
baselines and the full EG-IPG (report §Baselines, walkthrough §10.6).

Three levels of graph usage, each isolating one variable:

  1. NetworkPropagationBaseline   — topology-only diffusion of training responses over the PPI graph. No
                                    neural training, no evidence typing, no condition. Answers "how far does
                                    plain network smoothing get you?".
  2. UntypedGraphEncoder          — a homogeneous GCN over the protein graph with EVERY PPI edge collapsed
                                    to one untyped relation, no condition gate. Isolates topology learned by
                                    message passing, stripped of provenance (report's "untyped-graph
                                    diagnostic"). Trains via the Stage-A ``Trainer`` inside an ``EGIPGModel``.
  3. StaticTypedGraphEncoder      — the full ``TypedGraphEncoder`` with the condition gate PINNED to 1.0, so
                                    evidence types are kept but every edge counts equally regardless of
                                    culture condition. §10.6 nested-family member #2 (typed static graph);
                                    the isolated variable H2b removes.

The two neural encoders drop into ``EGIPGModel(graph_encoder=...)`` unchanged — they honour the same
``forward(target_genes, conditions, h_do) -> (h_graph, edge_gates, edge_confidences)`` contract the decoder
consumes, so screening trains and scores them through the identical Stage-A path.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import GATv2Conv, GCNConv

from tcell_pipeline import config
from tcell_pipeline.baselines.simple_baselines import BaseBaseline, _np
from tcell_pipeline.graph import (
    PROTEIN,
    TypedGraphEncoder,
    build_hetero_graph,
    sample_subgraph,
)
from tcell_pipeline.graph.graph_readout import GraphReadout
from tcell_pipeline.graph.typed_graph_encoder import _chunks

_PP_RELATIONS = ("physical_ppi", "co_complex", "functional_assoc")
_SCORE_COL = len(config.PPI_SOURCES)  # edge_attr layout: onehot(5) then score at index 5


# --------------------------------------------------------------------------------------------------
# 1. Network propagation (non-neural)
# --------------------------------------------------------------------------------------------------
class NetworkPropagationBaseline(BaseBaseline):
    """Diffuse training program-deltas over the symmetric-normalised PPI graph, then read the diffused
    field at each query target (Vanunu-style network propagation).

    Fit places each training target's mean Δz on its protein node (and a presence indicator on the same
    node), then propagates BOTH fields ``n_iter`` steps of ``F ← restart·S₀ + (1−restart)·Ŵ·F`` with the
    symmetric-normalised adjacency Ŵ = D^{-1/2} A D^{-1/2}. Predict returns ``F_signal[node] /
    F_presence[node]`` — a graph-proximity-weighted average of nearby training responses, so an unseen
    target inherits its neighbours' signal. Topology only: no evidence typing, no condition (a static
    smoother is exactly the point of this reference).
    ponytail: fixed ``n_iter`` power iterations instead of the exact ``(I − (1−r)Ŵ)^{-1}`` solve; raise
    ``n_iter`` (or swap in a sparse solve) if convergence on the real graph proves too slow."""

    def __init__(self, adjacency, gene_to_idx: dict[str, int], basis=None,
                 restart: float = 0.5, n_iter: int = 20) -> None:
        super().__init__(basis)
        self.gene_to_idx = gene_to_idx
        self.restart = float(restart)
        self.n_iter = int(n_iter)
        self._w = _sym_normalize(sp.csr_matrix(adjacency))
        self._n = self._w.shape[0]
        self._signal: np.ndarray | None = None   # (n_nodes, K) diffused signal
        self._presence: np.ndarray | None = None  # (n_nodes,) diffused presence

    @classmethod
    def from_hetero_graph(cls, graph=None, gene_to_idx: dict[str, int] | None = None, **kw
                          ) -> "NetworkPropagationBaseline":
        """Build the adjacency from a HeteroData PPI graph: union of the three protein-protein relations,
        each edge weighted by its source-confidence score, symmetrised. Any None loads from config paths."""
        if graph is None:
            graph, gene_to_idx = build_hetero_graph()
        gene_to_idx = gene_to_idx if gene_to_idx is not None else graph.gene_to_idx
        n = graph[PROTEIN].x.shape[0]
        rows, cols, wts = [], [], []
        for rel in _PP_RELATIONS:
            ei = graph[PROTEIN, rel, PROTEIN].edge_index
            ea = graph[PROTEIN, rel, PROTEIN].edge_attr
            if ei.numel() == 0:
                continue
            rows.append(ei[0].numpy())
            cols.append(ei[1].numpy())
            wts.append(ea[:, _SCORE_COL].numpy())
        if rows:
            r, c, w = np.concatenate(rows), np.concatenate(cols), np.concatenate(wts)
        else:
            r = c = w = np.zeros(0)
        a = sp.coo_matrix((w, (r, c)), shape=(n, n)).tocsr()
        a = a + a.T  # undirected
        return cls(a, gene_to_idx, **kw)

    def fit(self, genes, z, conditions=None) -> "NetworkPropagationBaseline":
        """genes: per-row target symbol; z: (M, K) program deltas. conditions is accepted for contract
        parity but ignored — network propagation is condition-agnostic topology smoothing."""
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
        s0[seen] /= counts[seen, None]                      # mean Δz per training-target node
        p0 = seen.astype(np.float64)                        # presence indicator
        self._signal = _propagate(self._w, s0, self.restart, self.n_iter)
        self._presence = _propagate(self._w, p0[:, None], self.restart, self.n_iter)[:, 0]
        return self

    def predict(self, genes, conditions=None) -> tuple[np.ndarray, np.ndarray]:
        if self._signal is None:
            raise RuntimeError("NetworkPropagationBaseline.predict called before fit")
        dz = np.zeros((len(genes), self._k))
        for i, g in enumerate(genes):
            j = self.gene_to_idx.get(g)
            if j is not None and self._presence[j] > 1e-12:
                dz[i] = self._signal[j] / self._presence[j]  # proximity-weighted mean of training responses
        return dz, self._decode_genes(dz)


def _sym_normalize(a: sp.csr_matrix) -> sp.csr_matrix:
    """Ŵ = D^{-1/2} A D^{-1/2}; isolated nodes (degree 0) get a zero row/column, so they neither send nor
    receive signal and fall back to a zero prediction."""
    deg = np.asarray(a.sum(1)).reshape(-1)
    with np.errstate(divide="ignore"):
        dinv = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    d = sp.diags(dinv)
    return (d @ a @ d).tocsr()


def _propagate(w: sp.csr_matrix, s0: np.ndarray, restart: float, n_iter: int) -> np.ndarray:
    f = s0.copy()
    for _ in range(n_iter):
        f = restart * s0 + (1.0 - restart) * (w @ f)
    return f


# --------------------------------------------------------------------------------------------------
# 2. Untyped homogeneous GCN
# --------------------------------------------------------------------------------------------------
class UntypedGraphEncoder(nn.Module):
    """Homogeneous GCN over the protein graph: every PPI edge (physical / co-complex / functional) is one
    untyped relation, edge provenance and the condition gate are discarded. Returns ``(h_graph, None,
    None)`` so an ``EGIPGModel`` wrapping it trains through the same decoder + Stage-A loss (the loss's
    graph-gate penalty is a no-op when ``edge_gates`` is None). ``conditions`` is ignored by design.

    Mini-batched like TypedGraphEncoder: the batch's subgraphs go through one PyG ``Batch`` so the GCN
    convolutions run once per batch rather than once per row.
    ponytail: the batch is still sampled row-by-row on CPU, which is now the throughput floor."""

    def __init__(self, graph=None, gene_to_idx: dict[str, int] | None = None,
                 hidden: int = config.GRAPH_HIDDEN_DIM, layers: int = config.GRAPH_LAYERS) -> None:
        super().__init__()
        if graph is None:
            graph, gene_to_idx = build_hetero_graph()
        self.graph = graph
        self.gene_to_idx = gene_to_idx if gene_to_idx is not None else graph.gene_to_idx
        self.hidden = hidden
        self.proj = nn.Linear(config.PROTEIN_FEATURE_DIM, hidden)
        self.convs = nn.ModuleList([GCNConv(hidden, hidden) for _ in range(layers)])
        self.readout = GraphReadout(hidden, config.GRAPH_N_HEADS)

    def _homogeneous_edges(self, sub, device) -> torch.Tensor:
        eis = [sub[PROTEIN, rel, PROTEIN].edge_index for rel in _PP_RELATIONS]
        present = [e for e in eis if e.numel()]
        ei = torch.cat(present, dim=1) if present else torch.zeros((2, 0), dtype=torch.long)
        return torch.cat([ei, ei.flip(0)], dim=1).to(device)  # undirected

    def _homogeneous_score(self, sub, device) -> torch.Tensor:
        """Per-edge STRING/source confidence aligned to ``_homogeneous_edges`` (same relation order, same
        undirected doubling), shape (E_doubled, 1). This is the signal the plain GCN throws away; the
        gat / wgcn variants feed it into attention / edge weights."""
        cols = [sub[PROTEIN, rel, PROTEIN].edge_attr[:, _SCORE_COL]
                for rel in _PP_RELATIONS if sub[PROTEIN, rel, PROTEIN].edge_index.numel()]
        if not cols:
            return torch.zeros((0, 1), device=device)
        s = torch.cat(cols).to(device).unsqueeze(-1)
        return torch.cat([s, s], dim=0)  # mirror the src||flip doubling in _homogeneous_edges

    def _message_pass(self, h, ei, sub, device) -> torch.Tensor:
        """The overridable convolution loop. Base = plain GCN, which ignores ``sub`` (no edge features).
        Subclasses that consume edge scores read them via ``_homogeneous_score(sub, device)``."""
        for conv in self.convs:
            h = F.relu(conv(h, ei))
        return h

    def encode_one(self, gene: str, h_do_row: torch.Tensor) -> torch.Tensor:
        device = self.proj.weight.device
        sub = sample_subgraph(self.graph, gene, gene_to_idx=self.gene_to_idx).to(device)
        h = F.relu(self.proj(sub[PROTEIN].x))
        ei = self._homogeneous_edges(sub, device)
        h = self._message_pass(h, ei, sub, device)
        h_graph, _ = self.readout(h_do_row.to(device).unsqueeze(0), h)
        return h_graph.squeeze(0)

    def forward(self, target_genes, conditions, h_do: torch.Tensor):
        device = self.proj.weight.device
        h_do = h_do.to(device)
        target_genes = list(target_genes)  # indexed by position below, where the old loop only zip()ed
        n = len(target_genes)
        known = [b for b, g in enumerate(target_genes) if g in self.gene_to_idx]
        if not known:  # no readout runs -> mirror h_do, the tensor the decoder concatenates this with
            return torch.zeros(n, self.hidden, device=device, dtype=h_do.dtype), None, None
        # at most GRAPH_ENCODE_CHUNK subgraphs in flight, so the caller's batch_size sets the
        # optimisation batch and not the memory ceiling (the per-row loop held exactly one)
        pooled = torch.cat([self._encode_chunk(part, target_genes, h_do)
                            for part in _chunks(known, config.GRAPH_ENCODE_CHUNK)], dim=0)
        # dtype follows the computed readout (as the old torch.stack did), not a hardcoded float32
        h_graph = torch.zeros(n, self.hidden, device=device, dtype=pooled.dtype)
        h_graph[torch.tensor(known, device=device)] = pooled
        return h_graph, None, None

    def _encode_chunk(self, part, target_genes, h_do: torch.Tensor) -> torch.Tensor:
        device = self.proj.weight.device
        subs = [sample_subgraph(self.graph, target_genes[b], gene_to_idx=self.gene_to_idx) for b in part]
        bat = Batch.from_data_list(subs).to(device)
        h = F.relu(self.proj(bat[PROTEIN].x))
        ei = self._homogeneous_edges(bat, device)
        h = self._message_pass(h, ei, bat, device)
        # each query attends over its own subgraph's nodes only (batch vector is already sorted)
        return self.readout(h_do[torch.tensor(part, device=device)], h, node_batch=bat[PROTEIN].batch)[0]


# --------------------------------------------------------------------------------------------------
# 2b. Augmented untyped encoders (AAAI stage-2): the two things GCN discards, added back
# --------------------------------------------------------------------------------------------------
class AugmentedUntypedEncoder(UntypedGraphEncoder):
    """``UntypedGraphEncoder`` with a choice of homogeneous convolution, to improve on the plain-GCN
    baseline by consuming the edge confidence GCN throws away:

      ``gcn``   the baseline: fixed symmetric ``1/sqrt(d_i d_j)``, edge scores unused (bit-identical to
                ``UntypedGraphEncoder``).
      ``gat``   ``GATv2Conv`` with ``edge_dim=1``: LEARNED attention weights that read the per-edge
                STRING score, instead of a fixed degree normalisation. Heads averaged (``concat=False``)
                so the hidden width is unchanged.
      ``wgcn``  ``GCNConv`` with ``edge_weight`` = the per-edge score: the cheapest way to let the edge
                confidence modulate the fixed normalisation.

    Untyped by construction — no condition gate, no relation types — so it keeps ``UntypedGraphEncoder``'s
    ``(h_graph, None, None)`` contract and trains through the same decoder / Stage-A loss."""

    CONVS = ("gcn", "gat", "wgcn")

    def __init__(self, graph=None, gene_to_idx=None, hidden: int = config.GRAPH_HIDDEN_DIM,
                 layers: int = config.GRAPH_LAYERS, *, conv: str = "gat",
                 heads: int = config.GRAPH_N_HEADS) -> None:
        if conv not in self.CONVS:
            raise ValueError(f"conv must be one of {self.CONVS}, got {conv!r}")
        super().__init__(graph, gene_to_idx, hidden, layers)  # builds proj / GCN convs / readout
        self.conv_kind = conv
        if conv == "gat":
            # concat=False averages the heads so out-dim stays `hidden`; edge_dim=1 lets the per-edge
            # score enter the attention logits (GATv2's edge-aware attention, Brody et al. 2021)
            self.convs = nn.ModuleList(
                [GATv2Conv(hidden, hidden, heads=heads, concat=False, edge_dim=1, add_self_loops=True)
                 for _ in range(layers)])
        # 'wgcn' reuses the base GCNConv modules; only the call in _message_pass differs

    def _message_pass(self, h, ei, sub, device) -> torch.Tensor:
        if self.conv_kind == "gcn":
            return super()._message_pass(h, ei, sub, device)
        score = self._homogeneous_score(sub, device)
        for conv in self.convs:
            if self.conv_kind == "gat":
                h = F.relu(conv(h, ei, edge_attr=score))
            else:  # wgcn
                h = F.relu(conv(h, ei, edge_weight=score.squeeze(-1)))
        return h


# --------------------------------------------------------------------------------------------------
# 3. Typed static graph (condition gate pinned to 1.0)
# --------------------------------------------------------------------------------------------------
class StaticTypedGraphEncoder(TypedGraphEncoder):
    """§10.6 nested-family member #2: the full typed encoder with the condition gate pinned to 1.0, so
    evidence types and topology are retained but every edge is weighted identically regardless of culture
    condition. Overriding only ``_gate`` reuses all of TypedGraphEncoder's signed, typed message passing;
    the returned ``edge_gates`` are all 1.0, which is the isolated variable H2b (condition gating) removes.
    The condition embedding is left in place but has no effect (its gradient is zero)."""

    def _gate(self, rel: str, edge_attr: torch.Tensor, h_cond: torch.Tensor, edge_batch=None) -> torch.Tensor:
        return edge_attr.new_ones((edge_attr.size(0), 1))  # every edge counts equally, all conditions alike


GRAPH_BASELINES: dict = {
    "network_propagation": NetworkPropagationBaseline,
    "untyped_gnn": UntypedGraphEncoder,
    "typed_static": StaticTypedGraphEncoder,
}
