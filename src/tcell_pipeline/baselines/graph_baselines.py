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


# --------------------------------------------------------------------------------------------------
# 4. Shared-weight typed graph (A1 diagnostic)
# --------------------------------------------------------------------------------------------------
class SharedWeightTypedGraphEncoder(StaticTypedGraphEncoder):
    """``typed_static`` with ONE ``_RelMessage`` tied across all four relations instead of one each.

    WHY. On the frozen fold at n=7 edge typing costs -0.0120 systema (7/7 seeds, survives Bonferroni and
    Holm) while the plain untyped GCN is the best graph arm at +0.0043. Two explanations are confounded
    inside that contrast: the relation PARTITION may be the wrong inductive bias, or typed message passing
    may simply carry 4x the message parameters over the same edges, making the damage capacity and nothing
    to do with evidence types. This arm holds the typed encoder fixed — signed messages, edge features,
    complex nodes, gate pinned to 1.0 — and removes only the per-relation multiplicity.

    WHAT IT DOES AND DOES NOT IDENTIFY. Under ``norm='add'`` (typed_static's setting) the layer computes
    ``sum_r sum_{u in N_r(v)} f_r(u)``; tying ``f_r = f`` makes that identically ``sum_{u in N(v)} f(u)``,
    so the partition stops affecting the aggregate at the same moment the parameters drop. The two are ONE
    intervention here, not two separable ones, and a difference against typed_static cannot be attributed
    to parameter count alone. Separating them needs a third arm that keeps per-relation parameters over a
    PERMUTED partition (same module count, same relation sizes, no evidence information); the size of
    (permuted - typed_static) is then the part attributable to the typing's information content.

    ponytail: the tie is applied after ``__init__`` builds four modules and discards three, rather than by
    subclassing ``_GraphLayer``. Same result, and it leaves the typed encoder untouched."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        for layer in self.layers:
            shared = next(iter(layer.rel.values()))
            layer.rel = nn.ModuleDict({rel: shared for rel in layer.rel})


# --------------------------------------------------------------------------------------------------
# 5. Permuted-relation typed graph (A1 diagnostic: the tie-breaker for the shared-weight arm)
# --------------------------------------------------------------------------------------------------
# splitmix64's multipliers. Wrap-around on uint64 IS the mixing, so overflow is intended everywhere.
_MIX = tuple(np.uint64(k) for k in (0xFF51AFD7ED558CCD, 0xC4CEB9FE1A85EC53, 0x9E3779B97F4A7C15,
                                    0xBF58476D1CE4E5B9, 0x94D049BB133111EB))


def _edge_hash(src: np.ndarray, dst: np.ndarray, seed: int) -> np.ndarray:
    """Deterministic uint64 hash of an UNDIRECTED protein-protein edge, mixed with ``seed``.

    Symmetric in (src, dst) so an edge hashes the same whichever orientation it is stored in, and a
    pure function of the GLOBAL node ids — which is what makes the permutation consistent: the same
    edge is relabelled the same way in every subgraph it appears in, so the permuted partition is one
    fixed alternative partition and not per-sample routing noise."""
    k1, k2, k3, k4, k5 = _MIX
    a = np.minimum(src, dst).astype(np.uint64)
    b = np.maximum(src, dst).astype(np.uint64)
    with np.errstate(over="ignore"):  # wrap-around IS the mixing, not an error
        x = (a * k1) ^ (b * k2) ^ (np.uint64(seed) * k3)
        x ^= x >> np.uint64(30)
        x *= k4
        x ^= x >> np.uint64(27)
        x *= k5
        x ^= x >> np.uint64(31)
    return x


class PermutedTypedGraphEncoder(StaticTypedGraphEncoder):
    """``typed_static`` with each protein-protein edge's RELATION LABEL randomly reassigned.

    WHY. ``SharedWeightTypedGraphEncoder`` removes the relation partition and the per-relation
    parameters in one intervention (see its docstring), so on its own it cannot say which of the two
    costs the graph its benefit. This arm holds the parameter count, the module count and the routing
    structure exactly at ``typed_static``'s, and destroys ONLY the correspondence between an edge's
    evidence class and the weight matrix that processes it. What remains of the -0.0120 after
    permuting is the part the typing's information content is responsible for.

    WHERE THE RELABELLING HAPPENS, and why it is not done to the graph. The sampler ranks candidate
    neighbours by relation (``_PRIORITY_BONUS`` gives physical and co-complex a 1e6 bonus over
    functional), so permuting the stored graph would change WHICH neighbours enter the subgraph and
    the arm would differ from typed_static in two ways at once. Relabelling happens after sampling,
    through the ``_sample`` hook, so the node set and the edge multiset are bit-identical to
    typed_static's and only the routing changes.

    EXACT GLOBAL COUNTS. Each relation keeps its original edge count across the whole graph: the two
    hash thresholds are read off the sorted hashes of every PP edge, so the assignment is a genuine
    permutation rather than a multinomial draw at the relation proportions. This matters under
    ``norm='add'``, where a relation's contribution to a node update scales with its degree."""

    def __init__(self, graph=None, gene_to_idx: dict[str, int] | None = None, *,
                 permute_seed: int = 0, **kwargs) -> None:
        super().__init__(graph, gene_to_idx, **kwargs)
        self.permute_seed = int(permute_seed)
        self._cuts = self._relation_cuts()

    def _relation_cuts(self) -> np.ndarray:
        """The two hash thresholds that split every PP edge in the FULL graph into the ORIGINAL
        per-relation counts."""
        hashes, counts = [], []
        for rel in _PP_RELATIONS:
            ei = self.graph[PROTEIN, rel, PROTEIN].edge_index
            counts.append(int(ei.shape[1]))
            if ei.numel():
                hashes.append(_edge_hash(ei[0].numpy(), ei[1].numpy(), self.permute_seed))
        if not hashes:
            return np.zeros(len(_PP_RELATIONS) - 1, dtype=np.uint64)
        ordered = np.sort(np.concatenate(hashes))
        cuts, acc = [], 0
        for c in counts[:-1]:
            acc += c
            cuts.append(ordered[acc - 1] if acc else np.uint64(0))
        return np.array(cuts, dtype=np.uint64)

    def _assign(self, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
        """Each edge's NEW relation index, from its global endpoints.

        ``side='left'`` counts the cuts strictly below the hash, which makes every cut the LAST hash of
        its own relation. ``'right'`` would push a boundary edge into the next relation and, where a
        relation is empty, two cuts coincide and a hash equal to them would skip a relation entirely —
        both show up as counts drifting by one (test_permuted_relations_preserve_every_relation_edge_
        count_globally caught exactly that)."""
        return np.searchsorted(self._cuts, _edge_hash(src, dst, self.permute_seed), side="left")

    def _sample(self, target_gene: str):
        sub = super()._sample(target_gene)  # sampled under the TRUE relations, see the class docstring
        stores = [sub[PROTEIN, rel, PROTEIN] for rel in _PP_RELATIONS]
        kept = [(s.edge_index, s.edge_attr) for s in stores if s.edge_index.numel()]
        if not kept:
            return sub
        ei = torch.cat([e for e, _ in kept], dim=1)
        ea = torch.cat([a for _, a in kept], dim=0)
        orig = sub[PROTEIN].orig_idx.numpy()  # subgraph node id -> GLOBAL node id, so the hash is global
        which = self._assign(orig[ei[0].numpy()], orig[ei[1].numpy()])
        for k, rel in enumerate(_PP_RELATIONS):
            mask = torch.from_numpy(which == k)
            sub[PROTEIN, rel, PROTEIN].edge_index = ei[:, mask]
            sub[PROTEIN, rel, PROTEIN].edge_attr = ea[mask]
        return sub


GRAPH_BASELINES: dict = {
    "network_propagation": NetworkPropagationBaseline,
    "untyped_gnn": UntypedGraphEncoder,
    "typed_static": StaticTypedGraphEncoder,
    "typed_shared": SharedWeightTypedGraphEncoder,
    "typed_permuted": PermutedTypedGraphEncoder,
}
