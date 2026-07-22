"""TypedGraphEncoder: 3-layer relational GNN with condition-gated signed message passing.

For each perturbation target we sample its neighbourhood, then run message passing where every
message is *signed* and *condition-gated*:

  signed message   m = tanh(W_sign h_u) * relu(W_mag h_u)   -- sign = activation/inhibition,
                                                                magnitude = strength
  condition gate   alpha = sigmoid(w_gate[h_cond || f_e])    -- the SAME edge is weighted
                                                                differently under Rest vs Stim
  aggregate        h_v = LayerNorm(h_v + FFN(sum_r sum_u alpha * (W_r m + U_r f_e)))

The gate depends only on the culture condition and the edge features (not on h_u), so it is
computed once and reused across all 3 layers, and returned as ``edge_gates`` for Module 4's
mechanism attribution. Readout cross-attends h_do over the final node states -> h_graph in R^256.
"""
from __future__ import annotations

import torch
from torch import nn
from torch_geometric.data import Batch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree, dropout_edge

from tcell_pipeline import config
from tcell_pipeline.graph.graph_builder import COMPLEX, PROTEIN, build_hetero_graph
from tcell_pipeline.graph.graph_readout import GraphReadout
from tcell_pipeline.graph.neighborhood_sampler import sample_subgraph

_PP_RELATIONS = ("physical_ppi", "co_complex", "functional_assoc")
_MEMBERSHIP = "complex_membership"
_COND_INDEX = {c: i for i, c in enumerate(config.CONDITIONS)}
# edge feature = source one-hot(len PPI_SOURCES) | score | is_direct | n_supporting; the score column
# (clipped to [0,1]) is the per-edge SOURCE CONFIDENCE the graph regulariser's unsourced term reads.
_SCORE_COL = len(config.PPI_SOURCES)


def _store_key(rel: str):
    return (PROTEIN, rel, PROTEIN) if rel in _PP_RELATIONS else (PROTEIN, _MEMBERSHIP, COMPLEX)


def _chunks(items: list, size: int):
    """Split into runs of at most ``size`` (0/None == one run: no chunking)."""
    size = size or len(items)
    return [items[i:i + size] for i in range(0, len(items), size)]


def signed_message(h: torch.Tensor, w_sign: nn.Linear, w_mag: nn.Linear) -> torch.Tensor:
    """m = tanh(W_sign h) * relu(W_mag h): tanh carries the sign, relu the (non-negative) magnitude."""
    return torch.tanh(w_sign(h)) * torch.relu(w_mag(h))


class _RelMessage(MessagePassing):
    """One relation's message + aggregation for one layer (custom message on PyG MessagePassing).

    ``norm`` selects how a node's incoming messages for THIS relation are combined:

      ``add``   plain sum (the original). A relation's contribution then scales with its degree, so
                the most abundant relation dominates the node update regardless of how informative it
                is. On the real graph ``functional_assoc`` is 86% of all edges at a median score of
                0.228, and the aggregate is summed again across relations — so the least reliable
                evidence class wins by sheer count.
      ``mean``  divide by the in-degree for this relation, so every relation contributes on the same
                scale and relative weight is set by the learned parameters (and the condition gate),
                not by annotation density.
      ``gcn``   symmetric ``1/sqrt(d_i d_j)``, the normalisation ``GCNConv`` applies. This is the
                isolated difference against ``UntypedGraphEncoder``, which uses ``GCNConv`` and is the
                BEST graph variant (+0.0045 vs no-graph) while this encoder is the worst (-0.0131).
    """

    NORMS = ("add", "mean", "gcn")

    def __init__(self, hidden: int, edge_dim: int, edge_dropout: float, norm: str = "add") -> None:
        if norm not in self.NORMS:
            raise ValueError(f"norm must be one of {self.NORMS}, got {norm!r}")
        super().__init__(aggr="mean" if norm == "mean" else "add")
        self.norm = norm
        self.w_r = nn.Linear(hidden, hidden)
        self.u_r = nn.Linear(edge_dim, hidden)
        self.w_sign = nn.Linear(hidden, hidden)
        self.w_mag = nn.Linear(hidden, hidden)
        self.p = edge_dropout

    def forward(self, x_src, x_dst, edge_index, edge_attr, alpha):
        ei, mask = dropout_edge(edge_index, p=self.p, training=self.training)
        w = None
        if self.norm == "gcn" and ei.numel():
            # symmetric 1/sqrt(d_src * d_dst), computed on THIS relation's degrees only
            d_src = degree(ei[0], x_src.size(0)).clamp(min=1)
            d_dst = degree(ei[1], x_dst.size(0)).clamp(min=1)
            w = (d_src[ei[0]] * d_dst[ei[1]]).rsqrt().unsqueeze(-1)
        return self.propagate(
            ei, x=(x_src, x_dst), edge_attr=edge_attr[mask], alpha=alpha[mask], w=w,
            size=(x_src.size(0), x_dst.size(0)),
        )

    def message(self, x_j, edge_attr, alpha, w):
        m = self.w_r(signed_message(x_j, self.w_sign, self.w_mag)) + self.u_r(edge_attr)
        m = alpha * m
        # `w` (the gcn degree weight) comes from torch_geometric.utils.degree, which is float32
        # regardless of the ambient autocast dtype; cast to the message dtype so the gcn path stays
        # autocast-clean by construction (else `w * m` promotes to float32 and only self-heals when a
        # later Linear happens to recast it — the precision-under-autocast fragility this repo has hit).
        return m if w is None else w.to(m.dtype) * m


class _FFN(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.norm = nn.LayerNorm(hidden)

    def forward(self, h: torch.Tensor, agg: torch.Tensor) -> torch.Tensor:
        return self.norm(h + self.net(agg))


class _GraphLayer(nn.Module):
    """One relational layer: message over 4 relations, residual FFN+LayerNorm per node type."""

    def __init__(self, hidden: int, edge_dim: int, edge_dropout: float, norm: str = "add",
                 rel_scale: bool = False) -> None:
        super().__init__()
        rels = (*_PP_RELATIONS, _MEMBERSHIP)
        self.rel = nn.ModuleDict({r: _RelMessage(hidden, edge_dim, edge_dropout, norm=norm) for r in rels})
        # One learnable scalar per relation, init 1.0 (identity at init, so this cannot shift a run on
        # its own). It lets the model down-weight a whole evidence class — the thing an unnormalised
        # `add` denies it, because there the weight is fixed by degree.
        self.scale = nn.ParameterDict(
            {r: nn.Parameter(torch.ones(())) for r in rels}) if rel_scale else None
        self.ffn_protein = _FFN(hidden)
        self.ffn_complex = _FFN(hidden)

    def _s(self, rel: str):
        return 1.0 if self.scale is None else self.scale[rel]

    def forward(self, h_p, h_c, edges):
        agg_p = torch.zeros_like(h_p)
        agg_c = torch.zeros_like(h_c)
        for rel in _PP_RELATIONS:
            ei, ea, al = edges[rel]
            if ei.numel():
                agg_p = agg_p + self._s(rel) * self.rel[rel](h_p, h_p, ei, ea, al)
        ei, ea, al = edges[_MEMBERSHIP]
        if ei.numel():
            s = self._s(_MEMBERSHIP)
            agg_c = agg_c + s * self.rel[_MEMBERSHIP](h_p, h_c, ei, ea, al)          # protein -> complex
            agg_p = agg_p + s * self.rel[_MEMBERSHIP](h_c, h_p, ei.flip(0), ea, al)  # complex -> protein
        return self.ffn_protein(h_p, agg_p), self.ffn_complex(h_c, agg_c)


class TypedGraphEncoder(nn.Module):
    def __init__(self, graph=None, gene_to_idx: dict[str, int] | None = None, *,
                 norm: str = "add", rel_scale: bool = False) -> None:
        """``norm`` / ``rel_scale`` are the aggregation-scale knobs (see ``_RelMessage``). Defaults
        reproduce the original encoder exactly, so an untouched call is bit-identical to before."""
        super().__init__()
        if graph is None:
            graph, gene_to_idx = build_hetero_graph()
        self.graph = graph
        self.gene_to_idx = gene_to_idx if gene_to_idx is not None else graph.gene_to_idx
        hidden, edge_dim = config.GRAPH_HIDDEN_DIM, config.EDGE_FEATURE_DIM

        self.proj = nn.Linear(config.PROTEIN_FEATURE_DIM, hidden)
        self.complex_embed = nn.Embedding(max(int(graph[COMPLEX].num_nodes), 1), config.COMPLEX_EMBED_DIM)
        self.condition = nn.Embedding(len(config.CONDITIONS), config.CONDITION_EMBED_DIM)
        # one condition gate per relation (alpha is layer-independent -> computed once, reused)
        self.gate = nn.ModuleDict(
            {r: nn.Linear(config.CONDITION_EMBED_DIM + edge_dim, 1) for r in (*_PP_RELATIONS, _MEMBERSHIP)}
        )
        self.layers = nn.ModuleList(
            [_GraphLayer(hidden, edge_dim, config.EDGE_DROPOUT, norm=norm, rel_scale=rel_scale)
             for _ in range(config.GRAPH_LAYERS)]
        )
        self.readout = GraphReadout(hidden, config.GRAPH_N_HEADS)

    def _edges_with_gates(self, sub, h_cond, device, edge_batch=None):
        """Compute the per-edge condition gate once per relation, then symmetrise PP edges for
        undirected message passing.

        The RETURNED ``gates[rel]`` is one value per ORIGINAL edge (length E, aligned to the
        sub-graph's ``edge_index``) for every relation — the mirrored second direction of a PP edge
        carries an identical gate (the gate reads only edge features + condition), so we don't double
        it into the Module-4-facing ``edge_gates``. ponytail: gate identity mapping (gate -> (u,v)) is
        recoverable from the sub-graph's edge_index; forward the full identity API with Module 4.

        ``edge_batch`` (dict relation -> per-edge sample id) is the mini-batched path, where ``sub``
        holds several targets' subgraphs and ``h_cond`` has one row per sample: each edge is then
        gated by ITS OWN sample's condition. None == one subgraph, one condition."""
        edges, gates, confs = {}, {}, {}
        for rel in _PP_RELATIONS:
            ei = sub[PROTEIN, rel, PROTEIN].edge_index
            ea = sub[PROTEIN, rel, PROTEIN].edge_attr
            alpha = self._gate(rel, ea, h_cond, edge_batch)           # E (one per original edge)
            edges[rel] = (torch.cat([ei, ei.flip(0)], dim=1),         # 2E: undirected message passing
                          torch.cat([ea, ea], dim=0),
                          torch.cat([alpha, alpha], dim=0))
            gates[rel] = alpha.squeeze(-1)                            # length E, consistent across relations
            confs[rel] = ea[:, _SCORE_COL]                            # per-edge source confidence, aligned to gates
        ei = sub[PROTEIN, _MEMBERSHIP, COMPLEX].edge_index
        ea = sub[PROTEIN, _MEMBERSHIP, COMPLEX].edge_attr
        alpha = self._gate(_MEMBERSHIP, ea, h_cond, edge_batch)
        edges[_MEMBERSHIP] = (ei, ea, alpha)
        gates[_MEMBERSHIP] = alpha.squeeze(-1)
        confs[_MEMBERSHIP] = ea[:, _SCORE_COL]
        return edges, gates, confs

    def _gate(self, rel: str, edge_attr: torch.Tensor, h_cond: torch.Tensor, edge_batch=None) -> torch.Tensor:
        """Per-edge condition gate. ``h_cond`` is (1, D) for a single subgraph; on the batched path it
        is (K, D) and ``edge_batch[rel]`` says which sample each edge belongs to, so every edge is
        gated by ITS OWN sample's condition. The per-edge expansion happens HERE rather than at the
        call site so an override that ignores the condition (StaticTypedGraphEncoder) doesn't pay for
        a (E, D) index_select it throws away."""
        if edge_attr.numel() == 0:
            return edge_attr.new_zeros((0, 1))
        cond = h_cond if edge_batch is None else h_cond[edge_batch[rel]]
        cond = cond.expand(edge_attr.size(0), -1) if cond.size(0) == 1 else cond
        return torch.sigmoid(self.gate[rel](torch.cat([cond, edge_attr], dim=1)))

    @staticmethod
    def _condition_index(condition) -> int:
        """Resolve a culture condition to its embedding row. Unknown conditions are invalid input
        (the vocab is closed — see config.CONDITIONS), so fail fast with a legible message rather than
        a cryptic KeyError/IndexError deep in the embedding lookup."""
        if isinstance(condition, str):
            if condition not in _COND_INDEX:
                raise ValueError(f"unknown culture_condition {condition!r}; valid: {list(_COND_INDEX)}")
            return _COND_INDEX[condition]
        idx = int(condition)
        if not 0 <= idx < len(config.CONDITIONS):
            raise ValueError(f"culture_condition index {idx} out of range [0, {len(config.CONDITIONS)})")
        return idx

    def encode_subgraph(self, sub, condition, h_do: torch.Tensor, keep_mask=None) -> dict:
        """Run message passing on an ALREADY-sampled subgraph, exposing the final-layer node states
        Module 4's rationale head reads. ``keep_mask`` (dict relation -> per-edge weight of length E,
        bool or float) scales that relation's condition gate, so the faithfulness deletion tests can
        re-run this frozen encoder with a rationale kept (weight on the selected edges) or removed
        (weight on the complement) — the gate multiplies every message, so a zero weight drops the
        edge at all layers. Returns ``{h_graph (dim,), gates, edge_confidences, node_states, attn (N,)}``
        where node_states is ``{protein: h_p, complex: h_c}`` and edge_confidences is the per-edge source
        confidence (the edge-feature score column, aligned to gates) the graph regulariser reads."""
        device = self.proj.weight.device
        h_do = h_do.to(device)  # public entry point: caller's h_do may be on a different device
        sub = sub.to(device)
        h_cond = self.condition(torch.tensor([self._condition_index(condition)], device=device))  # (1, 64)

        edges, gates, confs = self._edges_with_gates(sub, h_cond, device)
        if keep_mask is not None:
            edges = self._weight_edges(edges, keep_mask, device)
        h_p = self.proj(sub[PROTEIN].x)
        c_idx = sub[COMPLEX].orig_idx if sub[COMPLEX].num_nodes else torch.zeros(0, dtype=torch.long, device=device)
        h_c = self.complex_embed(c_idx)
        for layer in self.layers:
            h_p, h_c = layer(h_p, h_c, edges)

        h_graph, weights = self.readout(h_do.unsqueeze(0), torch.cat([h_p, h_c], dim=0))
        return {"h_graph": h_graph.squeeze(0), "gates": gates, "edge_confidences": confs,
                "node_states": {PROTEIN: h_p, COMPLEX: h_c}, "attn": weights.squeeze(0)}

    @staticmethod
    def _weight_edges(edges: dict, keep_mask: dict, device) -> dict:
        """Scale each relation's gate by a per-edge weight (Module 4 faithfulness masking). PP gates
        were mirrored to 2E for undirected passing, so the length-E weight is duplicated to match."""
        out = {}
        for rel, (ei, ea, alpha) in edges.items():
            w = keep_mask.get(rel)
            if w is None:
                out[rel] = (ei, ea, alpha)
                continue
            w = w.to(device=device, dtype=alpha.dtype).reshape(-1, 1)
            if rel in _PP_RELATIONS:  # alpha is 2E (both directions); the weight is E, so mirror it
                w = torch.cat([w, w], dim=0)
            out[rel] = (ei, ea, alpha * w)
        return out

    def encode_one(self, target_gene: str, condition, h_do: torch.Tensor):
        """Encode a single (target, condition) -> (h_graph (dim,), edge_gates dict, attn (N,))."""
        sub = sample_subgraph(self.graph, target_gene, gene_to_idx=self.gene_to_idx)
        r = self.encode_subgraph(sub, condition, h_do)
        return r["h_graph"], r["gates"], r["attn"]

    def forward(self, target_genes: list[str], conditions: list[str], h_do: torch.Tensor):
        """(B targets, B conditions, h_do (B, dim)) -> (h_graph (B, dim), edge_gates, edge_confidences).

        Each target has its own subgraph, so ``edge_gates[relation]`` / ``edge_confidences[relation]`` are
        lists over the batch (one per-edge tensor per sample) — the per-sample structure Module 4 and the
        Stage-A graph regulariser need; the two are aligned per edge. Unknown target genes fall back to a
        zero h_graph; an out-of-vocab culture_condition is invalid input and raises.

        Subgraphs are message-passed as PyG ``Batch``es: edges never cross samples (the batch offsets
        node ids), so one set of relational kernels replaces a per-row Python loop. The gate is
        scattered per edge and the readout attends per sample, so the result matches the per-row loop
        edge for edge (test_batched_forward_matches_per_sample_loop). At most
        ``config.GRAPH_ENCODE_CHUNK`` subgraphs go through message passing at once, so the caller's
        batch size picks the optimisation batch while peak memory stays bounded — the per-row loop this
        replaced held exactly one subgraph, and evaluation scores at BATCH_SIZE=64 under no_grad.
        ponytail: the batch is sampled row-by-row on CPU, so sampling is now the floor; make the
        sampler batch-aware (or cache subgraphs per target) if it becomes the bottleneck again.
        """
        device = self.proj.weight.device
        h_do = h_do.to(device)
        # materialise once: the batched path indexes by position, where the old loop only zip()ed, so
        # a generator/iterable caller would otherwise silently see an empty batch
        target_genes, conditions = list(target_genes), list(conditions)
        rels = (*_PP_RELATIONS, _MEMBERSHIP)
        n = len(target_genes)
        empty = torch.zeros(0, device=device, dtype=h_do.dtype)  # one shared blank, not 4n of them
        edge_gates = {r: [empty] * n for r in rels}
        edge_confidences = {r: [empty] * n for r in rels}
        known = [b for b, g in enumerate(target_genes) if g in self.gene_to_idx]
        if not known:
            # every target absent -> all-zero h_graph, empty gates. No readout runs, so there is no
            # computed dtype to follow; mirror h_do, which is what the decoder concatenates this with.
            return (torch.zeros(n, config.GRAPH_HIDDEN_DIM, device=device, dtype=h_do.dtype),
                    edge_gates, edge_confidences)

        pooled_parts = []
        for part in _chunks(known, config.GRAPH_ENCODE_CHUNK):
            pooled, gates, confs = self._encode_chunk(part, target_genes, conditions, h_do)
            pooled_parts.append(pooled)
            for r in rels:
                for i, b in enumerate(part):
                    edge_gates[r][b] = gates[r][i]
                    edge_confidences[r][b] = confs[r][i]
        pooled = torch.cat(pooled_parts, dim=0)
        # dtype follows what the readout actually produced, exactly as the old torch.stack(h_graphs)
        # did; a hardcoded-float32 buffer would silently downcast the result under .half()/autocast
        h_graph = torch.zeros(n, config.GRAPH_HIDDEN_DIM, device=device, dtype=pooled.dtype)
        h_graph[torch.tensor(known, device=device)] = pooled
        return h_graph, edge_gates, edge_confidences

    def _encode_chunk(self, part: list[int], target_genes, conditions, h_do):
        """Message-pass one chunk of the batch's known rows -> (pooled (k, dim), gates, confidences),
        the latter two as per-relation lists holding one per-edge tensor per row of ``part``."""
        device = self.proj.weight.device
        rels = (*_PP_RELATIONS, _MEMBERSHIP)
        subs = [sample_subgraph(self.graph, target_genes[b], gene_to_idx=self.gene_to_idx) for b in part]
        bat = Batch.from_data_list(subs).to(device)  # concatenate on CPU, then ONE host->device copy
        h_cond = self.condition(
            torch.tensor([self._condition_index(conditions[b]) for b in part], device=device)
        )  # (k, 64) — one condition row per kept sample
        p_batch = bat[PROTEIN].batch
        edge_batch = {r: p_batch[bat[_store_key(r)].edge_index[0]] for r in rels}  # each edge's sample

        edges, gates, confs = self._edges_with_gates(bat, h_cond, device, edge_batch)
        h_p = self.proj(bat[PROTEIN].x)
        h_c, c_batch = self._batched_complex_states(bat, p_batch, device)
        for layer in self.layers:
            h_p, h_c = layer(h_p, h_c, edges)

        # readout per sample: sort the concatenated protein+complex states by sample so each query
        # attends over its own subgraph only. The stable sort keeps proteins-then-complexes within a
        # sample, matching the single-subgraph cat([h_p, h_c]) order.
        node_batch = torch.cat([p_batch, c_batch])
        perm = torch.argsort(node_batch, stable=True)
        pooled = self.readout(
            h_do[torch.tensor(part, device=device)],
            torch.cat([h_p, h_c], dim=0)[perm], node_batch=node_batch[perm],
        )[0]

        out_g, out_c = {}, {}
        for r in rels:
            counts = [int(s[_store_key(r)].edge_index.shape[1]) for s in subs]
            # .clone(): torch.split returns VIEWS into the chunk-wide tensor, so without this an
            # in-place consumer (alpha.clamp_) would raise, holding one sample's gates would pin the
            # whole chunk's storage, and the chunk could never be freed. The per-sample loop this
            # replaced handed out independent tensors.
            out_g[r] = [g.clone() for g in torch.split(gates[r], counts)]
            out_c[r] = [c.clone() for c in torch.split(confs[r], counts)]
        return pooled, out_g, out_c

    def _batched_complex_states(self, bat, p_batch, device):
        """Complex embeddings for the batch + each complex's sample id. ``orig_idx`` survives batching
        un-offset (it is not an ``*index`` attribute), so it still points at the global complex table."""
        n_c = int(bat[COMPLEX].num_nodes or 0)
        if not n_c:  # no target in this batch belongs to any complex
            return self.complex_embed(torch.zeros(0, dtype=torch.long, device=device)), p_batch.new_zeros(0)
        return self.complex_embed(bat[COMPLEX].orig_idx), bat[COMPLEX].batch
