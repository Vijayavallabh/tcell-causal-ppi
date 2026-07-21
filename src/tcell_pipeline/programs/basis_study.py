"""feat-005 method x K basis CHARACTERISATION STUDY (§6.5) — scores candidate bases, changes nothing.

Every candidate is fit IN MEMORY and scored here; nothing in this module writes to
``data/intermediate/``. The frozen production basis (gene_program_loadings.parquet) is read-only
input, never an output — see docs/feat005-basis-study-notes.md.

Three axes, all defined against the frozen fit's recorded numbers (recon MAE 0.687 vs 0.817
zero-baseline, 22.7% exact-zero loadings for sparse_pca K=128):

* RECONSTRUCTION - MAE(Zc, A @ B.T) where Zc is the gene-wise-centred train matrix, against the
  predict-zero baseline MAE(Zc, 0). Centring matters: sparse_pca/fastica score their own centred
  target, so an uncentred target would not reproduce the recorded cell.
* SPARSITY - exact-zero fraction of B (no tolerance; the recorded 22.7% is exact zeros) + dead count.
* STABILITY - see ``matched_stability``. A basis is identified only up to permutation and sign, so
  components are matched by Hungarian assignment on |cosine| before being compared. An unmatched
  correlation is not a stability number.
"""
from __future__ import annotations

import numpy as np

_ROW_CHUNK = 2048  # keeps the dense residual off the heap on the real 21262 x 10282 matrix


def matched_stability(B1: np.ndarray, B2: np.ndarray) -> dict:
    """Mean |cosine| between components of two bases under an optimal one-to-one matching.

    Hungarian assignment (not greedy): each component is used exactly once, so a near-duplicate
    component cannot be double-counted into an inflated score. Sign is discarded via |cosine|
    because a factorisation is only identified up to a sign flip.

    Dead (all-zero) components have undefined cosine. They score 0 and are counted in ``n_dead``;
    if either basis is entirely dead there is nothing to match and ``mean_abs_cosine`` is None
    (UNDECIDABLE), never 0.0 — which would read as a real low-stability result.
    """
    from scipy.optimize import linear_sum_assignment

    B1 = np.asarray(B1, dtype=np.float64)
    B2 = np.asarray(B2, dtype=np.float64)
    if B1.ndim != 2 or B2.ndim != 2:
        raise ValueError(f"bases must be 2-D (G,K), got {B1.shape} and {B2.shape}")
    if B1.shape != B2.shape:
        raise ValueError(f"bases must share (G,K) to be matched, got {B1.shape} vs {B2.shape}")

    n1 = np.linalg.norm(B1, axis=0)
    n2 = np.linalg.norm(B2, axis=0)
    n_dead = int((n1 == 0).sum() + (n2 == 0).sum())
    if (n1 == 0).all() or (n2 == 0).all():
        return {"mean_abs_cosine": None, "n_dead": n_dead}

    U1 = B1 / np.where(n1 == 0, 1.0, n1)  # dead columns stay all-zero -> cosine 0 with everything
    U2 = B2 / np.where(n2 == 0, 1.0, n2)
    C = np.abs(U1.T @ U2)
    rows, cols = linear_sum_assignment(-C)
    return {"mean_abs_cosine": float(C[rows, cols].mean()), "n_dead": n_dead}


def recon_metrics(Zc: np.ndarray, A: np.ndarray, B: np.ndarray) -> dict:
    """Reconstruction of the centred target ``Zc`` by ``A @ B.T``, vs the predict-zero baseline."""
    Zc = np.asarray(Zc)
    row_mae = np.empty(Zc.shape[0], dtype=np.float64)
    abs_tgt = 0.0
    for i in range(0, Zc.shape[0], _ROW_CHUNK):
        z = Zc[i:i + _ROW_CHUNK].astype(np.float64)
        row_mae[i:i + _ROW_CHUNK] = np.abs(z - A[i:i + _ROW_CHUNK] @ B.T).mean(axis=1)
        abs_tgt += np.abs(z).sum()
    recon, base = float(row_mae.mean()), abs_tgt / Zc.size
    return {
        "recon_mae": recon,
        "zero_baseline_mae": base,
        # A target that is identically zero has nothing to explain — UNDECIDABLE, not 0% or 100%.
        "explained_frac": None if base == 0 else 1.0 - recon / base,
        "row_mae": row_mae,  # persisted per cell: the paired cell-vs-cell contrast needs it
    }


def sparsity_metrics(B: np.ndarray) -> dict:
    """Exact-zero fraction of the loadings + count of dead (all-zero) programs."""
    B = np.asarray(B)
    return {"zero_frac": float((B == 0).mean()), "n_dead": int((np.abs(B).sum(0) == 0).sum())}


def paired_recon_contrast(ref_row_mae: np.ndarray, cand_row_mae: np.ndarray) -> dict:
    """Paired t-test on per-row reconstruction MAE: candidate minus reference (negative == better).

    Zero-variance differences are UNDECIDABLE. A constant difference gives t = d/0 = inf and an
    identical pair gives 0/0; both once got reported in this project as ``p=0.0, CI excludes zero``,
    turning the one condition that proves the inputs carry no information into the strongest
    possible evidence. Both return ``p_raw = None``.
    """
    from scipy import stats

    d = np.asarray(cand_row_mae, dtype=np.float64) - np.asarray(ref_row_mae, dtype=np.float64)
    if d.size < 2:
        raise ValueError(f"need at least 2 paired rows, got {d.size}")
    mean = float(d.mean())
    sd = float(d.std(ddof=1))
    # Not `sd == 0`: a *constant* difference is only constant to float64 resolution, so sd lands at
    # ~1e-17 rather than 0, t = d/sd explodes, and p collapses to 0.0 — machine noise reported as
    # overwhelming evidence. Compare sd against the resolution of the values actually differenced.
    scale = max(float(np.abs(ref_row_mae).max()), float(np.abs(cand_row_mae).max()), 1.0)
    if sd <= 1e-12 * scale:
        return {"n": int(d.size), "mean_diff": mean, "p_raw": None, "p_underflow": False,
                "ci_low": None, "ci_high": None}
    se = sd / np.sqrt(d.size)
    half = stats.t.ppf(0.975, d.size - 1) * se
    p = float(stats.ttest_rel(cand_row_mae, ref_row_mae).pvalue)
    return {
        "n": int(d.size), "mean_diff": mean, "p_raw": p,
        # p == 0.0 is a double-precision underflow, not certainty. At this n it happens for
        # differences far too small to matter, so the flag travels with the number.
        "p_underflow": p == 0.0,
        "ci_low": mean - half, "ci_high": mean + half,
    }


def multiplicity_adjust(p_values: list) -> dict:
    """Bonferroni AND Holm over the family. Both are always reported.

    Recording only one invites picking the method that rescues the claim after seeing the numbers —
    the look-elsewhere effect in a lab coat. ``None`` entries are undecidable tests: they stay None
    and do not count toward the family size, because the family is the tests actually run.
    """
    idx = [i for i, p in enumerate(p_values) if p is not None]
    m = len(idx)
    bonf: list = [None] * len(p_values)
    holm: list = [None] * len(p_values)
    for i in idx:
        bonf[i] = min(1.0, m * p_values[i])
    running = 0.0
    for rank, i in enumerate(sorted(idx, key=lambda j: p_values[j])):
        running = max(running, min(1.0, (m - rank) * p_values[i]))  # step-down, kept monotone
        holm[i] = running
    return {"bonferroni": bonf, "holm": holm, "family_size": m}


def fit_vae_basis(Zc: np.ndarray, K: int, seed: int, epochs: int, batch_size: int = 256,
                  lr: float = 1e-3, beta: float = 1.0) -> tuple[np.ndarray, np.ndarray, dict]:
    """Shallow (single-linear-layer) VAE basis: Zc ~= A @ B.T with B the decoder weight.

    Deliberately linear so the result is a BASIS comparable to the matrix factorisations — a
    nonlinear decoder would not define a B at all. Returns (B (G,K), A (N,K) posterior means, info).
    """
    import torch

    torch.manual_seed(seed)
    X = torch.as_tensor(np.asarray(Zc, dtype=np.float32))
    N, G = X.shape
    enc = torch.nn.Linear(G, 2 * K)
    dec = torch.nn.Linear(K, G, bias=False)  # no bias: Zc is already centred, so B alone defines the basis
    opt = torch.optim.Adam([*enc.parameters(), *dec.parameters()], lr=lr)
    gen = torch.Generator().manual_seed(seed)  # seeded shuffling + reparam noise => reproducible B

    final_loss = float("nan")
    for _ in range(epochs):
        total = 0.0
        for idx in torch.randperm(N, generator=gen).split(batch_size):
            xb = X[idx]
            mu, logvar = enc(xb).chunk(2, dim=1)
            logvar = logvar.clamp(-10.0, 10.0)  # exp() of an unclamped logvar overflows to inf/NaN
            z = mu + torch.exp(0.5 * logvar) * torch.randn(mu.shape, generator=gen)
            recon = ((dec(z) - xb) ** 2).sum(1).mean()
            kl = (-0.5 * (1 + logvar - mu**2 - logvar.exp()).sum(1)).mean()
            loss = recon + beta * kl
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(idx)
        final_loss = total / N

    with torch.no_grad():
        A = torch.cat([enc(X[i:i + _ROW_CHUNK]).chunk(2, dim=1)[0] for i in range(0, N, _ROW_CHUNK)])
        B = dec.weight.detach().clone()  # Linear(K,G).weight is (G,K) == B
    return B.numpy(), A.numpy(), {"epochs_run": epochs, "final_loss": final_loss}
