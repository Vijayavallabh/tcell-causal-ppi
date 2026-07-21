# H1 optimisation notes — Phase 1 bug hunt

**Date:** 2026-07-21 · **Scope:** Phase 1 only (diagnosis, zero production edits) · **Probe:**
`src/tcell_pipeline/probe_graph_gradients.py` (new, read-only) · **Raw:** rerun with `--out probe.json`

Reproduce (exits non-zero if the probe's own self-control fails, or if the checkpoint it loads is not the
one `promoted.json` records as the frozen H1):

```bash
OMP_NUM_THREADS=4 PYTHONPATH=src uv run python -m tcell_pipeline.probe_graph_gradients \
    --n-max 32 --batch-size 4 --steps 6
```

**CORRECTION (2026-07-21), after a five-reviewer code review of this session's work.** Four numbers in
the first version of this document were wrong or overstated; all are fixed above and none change the
conclusion. (1) The Phase-2 timing basis quoted `probe_b`'s 11.5 s as "one true end-to-end Stage-A step";
`probe_b` omits the donor-invariance re-forwards, so that was a sub-component benchmark — the real basis
is 0.895 s/step through `Trainer._epoch` (§limitations). (2) §3 claimed the message-passing layers receive
gradient "only" from the response term; the gene term reaches them too. (3) §3's gradient range "1e-3 to
1e-7" was wrong — it is 9.69e-08 to 1.23e-01. (4) §8's sensitivity ratio was 119×, measured with DropEdge
live and non-reproducible; the deterministic eval-mode figure is 106×. §4's clip-scale chain was a
derivation and is now a measurement.

**Operational note — the GPU index you check is not the GPU you get.** `nvidia-smi` orders by PCI bus
id; torch's default CUDA enumeration does not. On this box `--device cuda:4` resolves to the **T400 4 GB**
while `nvidia-smi` calls that card index 3, so a Stage-1 lane dispatched to "the free A100 at index 4"
OOMed six minutes in, *after* the 34 s dataset build. Export `CUDA_DEVICE_ORDER=PCI_BUS_ID`, pin with
`CUDA_VISIBLE_DEVICES`, and confirm with `torch.cuda.get_device_properties(i).name` — never the index
alone. `pilot_lambda_graph.py` now **refuses to start** on any card under 16 GiB, printing the resolved
device name and the fix; that guard protects a session that never reads this file, which documentation
cannot.

**CORRECTION 2 (2026-07-21), found while smoke-testing the Stage-1 pilot.** §4 originally explained the
gate collapse as gradient-clipping "crowd-out" — the penalty's huge norm eating the step budget. That
mechanism is wrong: **AdamW is scale-invariant per parameter**, so a uniform clip factor cancels out of
the update entirely (demonstrated in §4). The measurements were right; the causal story was not. The
correct mechanism is *direction* dominance, not magnitude, and it carries a falsifiable prediction that
normalising `L_graph` by `|E|` will NOT rescue the gates — which the Stage-1 pilot is running now. The
conclusion (the penalty, not the data, determines the gates) is unchanged and, if anything, sharper.

---

## The question, and the answer

> Do the typed graph encoder's parameters receive non-zero gradients, and do they move from init?

**Yes to both. The gradient path is NOT severed — the pre-registered hypothesis is falsified.**

**But the negative was still measured on a model whose graph is switched off.** The mechanism is the
opposite of a severed wire: the gradient reaching the edge gates is *overwhelming*, and it points at
zero. `L_graph` annihilates the condition gate inside the first epoch, after which message passing is
multiplied by ~0 and the encoder degenerates to a per-node MLP over the target's own features.

Both halves matter. "Not severed" does not mean "the measurement was valid".

---

## Evidence

### 0. The probe's own falsification test (run first)

A deliberately severed encoder (`h_graph` detached before the decoder) must report no gradient. It did:
**148/148 graph parameters with `grad=None`, 0 with `‖grad‖>0`.** The probe can see severance, so its
failure to see severance below is a measurement, not a blind spot.

### 1. Not severed (probe A)

One real Stage-A forward+backward, `condition_gated`, real graph, real batch:

**0 of 148 graph-encoder parameters have `grad=None` or `‖grad‖==0` under the total loss.**

### 2. The loss is 103× the task at initialisation (probe A)

| term | raw | × lambda | contribution to total |
|---|---|---|---|
| response | 3.23 | 1.0 | **3.23** |
| gene | 0.4365 | 0.5 | 0.218 |
| de | 0.474 | 0.1 | 0.047 |
| invariance | 0 | 0.1 | 0 |
| **graph** | **3.328e+04** | 0.01 | **332.8** |

`graph penalty / response = 103.02×`. Against the fuller task side (response + gene + de = 3.496, since
the gene term also reaches the encoder — see §3) it is **95.20×**. Either way the penalty is ~100× the
task. `L_graph` is an unnormalised `sum_{e in E}` over **142,779 edges** in a 4-row batch. The formula is
faithful to the spec (walkthrough §8.2 item 5) — nobody sized `LAMBDA_GRAPH=0.01` against a real subgraph
of 16k–60k edges.

### 3. The gate gradient is ~10⁶× the response gradient (probe A)

Per-parameter decomposition — the same backward, split by source term:

| parameter | ‖g from response‖ | ‖g from graph penalty‖ | ratio |
|---|---|---|---|
| `gate.functional_assoc.bias` | 3.89e-05 | 1.22e+02 | **3,122,711×** |
| `gate.functional_assoc.weight` | 8.75e-04 | 6.82e+02 | **778,882×** |
| `condition.weight` | 1.50e-04 | 4.62e+01 | **308,941×** |
| `gate.physical_ppi.weight` | 4.85e-04 | 4.54e+01 | **93,521×** |
| `gate.complex_membership.weight` | 2.00e-03 | 1.45e+01 | **7,267×** |

The penalty reaches **only** the gates and the condition embedding — `_graph` is a pure function of
`edge_gates`/`edge_confidences`, so `g_graph_penalty` is `None` for all 139 message-passing parameters.
Those get their gradient from the response term **and from the gene term**: `program_decoder.py:66-68`
adds `residual(h_graph)` to `delta_x`, and `delta_z` mixes `graph_path([h_graph ‖ h_do])`, so the
gene-level Huber loss flows back through message passing too. The probe isolates `total`/`response`/`graph`
but not `gene`, and the gap shows: `g_total` exceeds `g_response` by −5.3% to +20.6% on exactly those 139
parameters. Their `g_response` spans **9.69e-08 to 1.23e-01** (48 of 139 above 1e-3) — still three to
seven orders below the gate gradients of 14.5–682, which is the point, but the spread is wide and the
smallest values are the deepest relations, whose messages are already being multiplied by a gate under
sentence of death.

At the gates themselves the omission is immaterial: `g_total / g_graph_penalty` is 0.999994–1.000315, so
the task side is undetectable there at any precision this probe reports.

**The prediction task had no vote on the gates.** They were not zeroed because the model discovered the
graph is useless; they were zeroed by a penalty whose gradient is a million times louder.

### 4. Gradient clipping converts domination into crowd-out (measured in probe B)

`probe_b` now captures `clip_grad_norm_`'s return value — the true pre-clip norm over **model + loss**
parameters, which is the denominator the whole update is actually divided by:

| quantity | value |
|---|---|
| measured pre-clip ‖g‖, step 0 | **695.53** |
| measured pre-clip ‖g‖, range over 6 steps | 455.10 – 1048.73 |
| `GRAD_CLIP = 1.0` → measured clip scale | **9.54e-04 – 2.20e-03** |
| graph-encoder's own share of that norm | **100.00%** (695.53 of 695.53) |
| ‖g‖ from the response term alone | 0.1718 |

The last row is worth stating plainly: the graph encoder's own gradient accounts for 100.00% of the
whole-model norm to five significant figures. The perturbation encoder, decoder and DE head contribute
nothing detectable in quadrature — the clip is set entirely by the gate penalty.

**But the magnitude is NOT the mechanism, and an earlier version of this section said it was.** It
claimed clipping "rescales the whole update by 1/695, so ~99.9% of the gradient budget is spent driving
gates to zero." That is wrong: **AdamW is scale-invariant per parameter.** Scaling every gradient by a
constant `c` scales the first moment by `c` and the second by `c²`, so `m̂/√v̂` is unchanged — and a
uniform clip factor is exactly such a constant. Verified numerically on one parameter under the real
settings (`lr=1e-3`, `wd=1e-5`, 300 steps):

| setup | resulting θ |
|---|---|
| gradient 1e-4 | −0.299969 |
| gradient 1e-1 (1000× larger) | −0.300000 |
| ‖g‖=695 **clipped to 1.0** | −0.300000 |
| ‖g‖=0.17 unclipped | −0.300000 |

The clip changes nothing. **What decides the gates is the gradient's DIRECTION**, and at the gate
parameters that direction is ~100% the penalty's (`g_total/g_penalty` = 0.999994–1.000315, §3). AdamW
then marches them at ~`lr` per step, in the same direction, for all 2,127 steps of every epoch — which
saturates the sigmoid long before epoch 0 ends. The prediction task does not lose a tug-of-war over step
size; it has no say in the direction at all.

This distinction has a sharp, falsifiable consequence. Since only the ratio's *dominance* matters and not
its size, shrinking the penalty does not free the gates unless it stops dominating:

| penalty/task ratio | θ after 300 steps |
|---|---|
| 1e+06 (measured, current) | −0.30000 |
| 1e+01 (what `L_graph / \|E\|` would give) | −0.30000 — **identical** |
| 0 (`lambda_graph=0`) | **+0.30000** — reverses |

So normalising `L_graph` by edge count, which cuts the ratio from ~10⁶ to ~10, is predicted **not** to
rescue the gates. Only removing the penalty is. The Stage-1 pilot tests exactly this.

### 5. Everything moves; nothing is stuck (probe B)

6 real AdamW steps (`lr=1e-3`, `weight_decay=1e-5`, `GRAD_CLIP=1.0` — Trainer's settings):

- **0 of 148 parameters failed to move.** No optimiser/param-group bug.
- `gate.physical_ppi.bias` drifted **77.7% of its init norm in six steps**.
- gate mean on the fixed probe batch: 0.6599 → 0.6129 (−7.1%) in six steps.

Train split is **21,262 rows = 2,658 steps/epoch at batch 8**. A 7% gate decay per 6 steps over 2,658
steps is annihilation well inside epoch 0 — which is exactly what the recorded histories show.

*(Ignore the `rel≈1e10` drift rows for LayerNorm/attention biases: those initialise at zero, so relative
drift divides by ~0. `gate.*.bias` initialises non-zero, so its 0.777 is real.)*

### 6. The decoder is NOT insensitive to h_graph (probe C)

| perturbation of h_graph | relative move in delta_z |
|---|---|
| eps = 1e-3 | 6.19e-04 |
| eps = 1e-1 | 7.40e-02 |
| eps = 1.0 | 6.95e-01 |

`‖d(sum delta_z²)/d(h_graph)‖ = 4.28`. Finding 1 is **not** a decoder-collapse story — consistent with the
already-refuted `lambda = 0.263`. The decoder is listening; h_graph has nothing to say.

### 7. Finding 3 resolved: typed_static's constant graph loss is BY DESIGN (probe D)

`StaticTypedGraphEncoder._gate` (`baselines/graph_baselines.py:229`) returns `edge_attr.new_ones(...)` —
a literal constant. Measured:

- graph term = 55916.4922, **`requires_grad=False`, `grad_fn=None`**
- gates: n=142,779, min = max = **exactly 1.000000**, `requires_grad=False`
- gradient on encoder parameters from the graph term alone: **NONE — the term is constant w.r.t. all 148**

Constructed inputs, per the "what would make this FIRE?" rule:

| constructed input | graph term | |
|---|---|---|
| baseline | 55916.4922 | |
| drop half the edges | 27982.6211 | **MOVED** |
| set every source confidence to 1 | 35694.7500 | **MOVED** |
| **randomise all 148 encoder parameters** | **55916.4922** | **unchanged** |

The term responds to the data and is provably invariant to every parameter. **By design, not severed.**

Corollary already flagged and now confirmed by arithmetic: `typed_static`'s `best_val = 490.420` is
`3.242 + 0.5(0.449) + 0.1(0.1209) + 0.01(48694.242) = 490.421`. It is ~99.3% a data constant.
**`best_val` is not comparable across arms and never was.** (Early stopping is unaffected — a constant
offset does not move an argmin.)

### 8. The decisive measurement: h_graph's dependence on the neighbourhood (probe E)

Target `A1BG`, condition `Rest`, 33,754 edges. Every edge gated off via the `keep_mask` path Module 4's
faithfulness tests use, so the target's own node features survive and only the neighbourhood dies:

| model | gate mean | delete-all-edges → relative change in h_graph |
|---|---|---|
| at init | 0.613755 | **1.1784** (the neighbourhood is ~all of it) |
| **frozen H1 checkpoint** | **0.000000** | **1.1142e-02** |

**Gate mean fell 4.5e+06×. Neighbourhood sensitivity fell 106×.** This is the causal link from the
penalty (§3–4) to session B's "delete the whole neighbourhood, prediction moves 1.8e-07 to 4.6e-05 of its
magnitude" (`feat008-stage-b-notes.md`, 8 val genes). Session B recorded the *what* and explicitly left
the *why* open. This is the why.

Measured with the model forced to `eval()`. That matters: `config.EDGE_DROPOUT` gates
`dropout_edge(training=self.training)`, and a freshly built module defaults to training mode, so an
earlier train-mode version of this probe reported 119× and varied run to run (1.1935 / 1.1966 / 1.1934
across three repeats). 106× is the deterministic figure.

The checkpoint is verified against the promotion registry rather than assumed from its path: the probe
reports `checkpoint IS the promoted H1; trained to epoch 1` and exits non-zero if `promoted.json` records
a different file. "Trained to epoch 1" is an independent corroboration of Finding 2's one-epoch model.

### 9. It is all five seeds, not one curve

| seed | graph term, end of epoch 0 | end of run | best epoch |
|---|---|---|---|
| 0 | 0.0189 | 0.000000 | 1 |
| 1 | 0.0226 | 0.000000 | 0 |
| 2 | 0.0229 | 0.000000 | 0 |
| 3 | 0.0214 | 0.000000 | 1 |
| 4 | 0.0241 | 0.000000 | 1 |

From 33,281 at init to ~0.02 by the end of epoch 0 — a factor of ~1.6e+06 — in every seed. And
`typed_static`'s graph term is **48694.242 in all five seeds, spread 0.00e+00**: bit-identical, as a
constant with no parameter dependence must be.

---

## What this means for the family — the asymmetry

`L_graph` only bites an arm that has *learnable* gates. It does not treat the four arms alike:

| arm | edge_gates | graph penalty | does its graph function? |
|---|---|---|---|
| `expression_only` | — (no encoder) | 0.000 | n/a |
| `untyped_gnn` | returns `None` | **no-op** (0.000 all 20 epochs) | **YES — fully on** |
| `typed_static` | pinned to 1.0, no grad | **constant, zero gradient** | **YES — fully on** |
| `condition_gated` **(= the frozen H1)** | learned | **103× the task loss** | **NO — annihilated in epoch 0** |

**The one arm whose graph is switched off is the frozen H1.** So:

- **H2b (`condition_gated` vs `typed_static`) is confounded.** Its isolated variable is supposed to be
  condition gating. It is actually "graph switched off by its own regulariser" vs "graph at full
  strength". That contrast does not test what it claims to test.
- **Every claim resting on the frozen H1's graph behaviour is confounded**, including the graph-independence
  result and any rationale/faithfulness verdict computed on that checkpoint.
- **Finding 2 is explained.** By epoch 1 `condition_gated` is expression-only plus a large inert
  graph-shaped parameter block; it then overfits (val.response 3.236 → 3.372) while `expression_only`
  plateaus (3.239 → 3.235). The "one-epoch model" is a model whose graph died at the end of epoch 0.

## What this does NOT license

**It does not say the graph would win if fixed. The evidence points both ways and must be reported that way.**

Among the arms whose graph actually functions, one beats and one trails the no-graph arm:

| arm | systema | graph functioning? |
|---|---|---|
| `untyped_gnn` | **+0.0951** | yes |
| `expression_only` | +0.0861 | no graph |
| `condition_gated` (H1) | +0.0834 | **no — annihilated** |
| `typed_static` | **+0.0786** | yes |

A working `condition_gated` could land anywhere in that spread. Specifically this diagnosis does **not**:

- rescue the graph hypothesis — it invalidates one arm's *test*, it does not supply a positive result;
- invalidate `expression_only` vs `untyped_gnn`, which is uncontaminated (neither has learnable gates);
- justify touching `promoted.json`, which stays frozen. Any repaired model is a NEW CANDIDATE owing the
  full protocol: same fold, 5 seeds, paired contrasts, Bonferroni **and** Holm;
- warrant fixing only the graph arm. Any change is applied to **all four** and the family is re-screened.
  The no-graph arm may gain as much — and if it does, that will be reported.

## Probe limitations, stated

- Probes A–D: 4 real rows, seed 0, CPU. The cross-seed val curves (§9, n=5) carry the collapse claim;
  the single-batch probes carry the gradient-decomposition claim.
- Probe A stays in **train mode on purpose** — DropEdge is part of the training step it diagnoses — but
  its forward is seeded, so the per-parameter ratios above reproduce bit-identically across runs.
- Probe E queries the readout with an **untrained** `h_do` against a trained graph encoder. Impact is
  smaller than it sounds: `h_do` enters the graph encoder *only* at the readout cross-attention, gates
  depend on condition + edge features alone, and message passing never sees `h_do` — so the small frozen
  `rel_delta` is driven by the (h_do-independent) gate collapse, not by the out-of-distribution query.
  `gate_mean = 0.000000` is exact. The 106× ratio is h_graph sensitivity, **not** end-to-end prediction
  sensitivity (session B measured that separately at 1.8e-07 – 4.6e-05).
- Probe D perturbs parameters in place on a throwaway model; nothing is persisted.
- **Timing: `probe_b` is NOT a benchmark and must not be quoted as one.** Its loop does one forward per
  step, while the real `Trainer._epoch` also calls `_donor_variants`, re-forwarding the whole model
  `DONOR_INVARIANCE_SAMPLES=2` more times per training batch. An earlier version of this document quoted
  `probe_b`'s 11.5 s as "one true end-to-end Stage-A step" — that was a sub-component benchmark passed
  off as a whole-pipeline estimate, the exact substitution AGENTS.md records. The field is now named
  `mean_step_seconds_EXCLUDING_donor_invariance` so it cannot be misread.
  **The real basis, for any Phase-2 estimate: 0.895 s/step** — measured post-warm-up through the actual
  `Trainer._epoch` (batch 8, A100, `donor_invariance_on=True`), and corroborated against the recorded
  campaign's 0.681 h/epoch for this arm (8.17 GPU-h / 12 epochs), agreeing within 3%.

## Stage-1 pilot result (2026-07-21) — mechanism confirmed, repair does not help

Three runs of `condition_gated`, seed 0, 2 epochs, batch 8, **identical in every respect except
`lambda_graph`'s scaling**. Trained on a TARGET-GROUPED inner split of the train fold (17,009 / 4,253
rows, 5,766 / 1,442 genes, zero gene overlap). **The val fold was not touched**; every number below is a
training diagnostic or an inner-holdout loss. Probe: `src/tcell_pipeline/pilot_lambda_graph.py`.

| arm | role | `lambda_graph` | gate mean init → ep1 | dead gates | neighbourhood sensitivity | verdict |
|---|---|---|---|---|---|---|
| `baseline` | status quo (what was screened) | 0.01, per sample | 0.678556 → **0.000000** | 100.0% | 9.41e-04 | COLLAPSED |
| `zero` | clean control | **0** | 0.678556 → **0.897950** | 0.7% | **9.03e-01** | SURVIVED |
| `normalised` | candidate repair | 0.01, **per edge** | 0.678556 → **0.000322** | 98.2% | 4.06e-03 | COLLAPSED |

**1. The collapse is caused by `L_graph`, confirmed prospectively.** Remove the penalty and the gates do
not merely survive — they rise to 0.90, and `h_graph` goes from neighbourhood-blind to almost entirely
neighbourhood-determined (**960× more sensitive** than baseline). The pre-registered kill criterion
("if gates still collapse at `lambda_graph=0`, the diagnosis is wrong") was not met.

**2. Normalising by |E| does NOT rescue the gates — as predicted before the run.** §4's direction-dominance
account predicted this: dividing by ~35,000 edges takes the penalty/task ratio from ~10⁶ to ~10, and a
ratio of 10 still owns the gradient's direction. Observed exactly: 71.2% of gates dead by epoch 0, 98.2%
by epoch 1. The **endpoint** matches the prediction; the **rate** differs (normalised takes ~1 epoch
longer), which the constant-gradient single-parameter model in §4 does not capture — Adam's moment
estimates take time to align when the ratio is small. Magnitude sets the *speed* of the collapse;
direction sets *whether* it happens.

**3. And the graph does not help, whether or not it works.** Inner-holdout response at epoch 1:

| arm | holdout response | neighbourhood sensitivity |
|---|---|---|
| `baseline` (graph dead) | **3.3545** | 9.41e-04 |
| `normalised` (graph dead) | 3.3554 | 4.06e-03 |
| `zero` (graph fully working) | 3.3568 | 9.03e-01 |

A **960× swing in graph dependence** moves the held-out response loss by 0.0023 — 0.07%, with the
working-graph arm marginally *last*. At n=1 seed over 2 epochs that spread is not a difference; the
honest statement is that the three are **indistinguishable**. Repairing the graph removes the excuse
without changing the answer.

**Scope, honestly:** 1 seed, 2 epochs, response/gene loss on an inner holdout — **not** the locked
`systema` endpoint, which cannot be read without opening val. This licenses "the negative was measured on
a non-functioning graph" and "the obvious repair does not overturn it". It does **not** license promoting
anything: any repaired model is a NEW CANDIDATE owing the full protocol (same fold, 5 seeds, paired
contrasts, Bonferroni AND Holm). `promoted.json` remains frozen and untouched.

## Recommended next step (Phase 2 — NOT started, needs approval)

The discriminating experiment is a **lambda_graph sensitivity sweep applied symmetrically to all four
arms**, with the gate trajectory logged per epoch. `lambda_graph = 0` is the clean control: it tells us
whether a `condition_gated` model that keeps its gates beats, matches, or trails no-graph — the
comparison the screening campaign intended to make and did not.

Two structural defects are separable from that sweep and worth deciding on their own merits:

1. `L_graph` scales with **edge count** (a data property, 16k–60k per row), not with a modelling choice.
   Normalising by `|E|` would make `lambda_graph` mean the same thing for a hub and a leaf. This is a
   change to a **spec'd formula**, so it is a decision to be taken, not a bug to be quietly fixed.
2. `best_val` is not comparable across arms whenever a constant loss term differs between them (§7).
   That is a reporting defect independent of everything above.

No file outside this document and the new probe has been touched. `promoted.json`, `feature_list.json`,
`progress.md`, and `session-handoff.md` are unmodified.
