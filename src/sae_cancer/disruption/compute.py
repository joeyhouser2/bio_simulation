"""Compute per-variant SAE feature-disruption vectors (brief §5.1.4).

For a missense variant at (1-indexed) protein position ``p``:

    f_wt, f_mut = SAE features [L+2, codebook] for the WT and mutant sequences
                  (token index 0 = <cls>, p = residue p, last = <eos>)

    local_delta    = f_mut[p] - f_wt[p]                 # signed, at the mutated residue
    window_absdelta= sum_{j in p±w} |f_mut[j] - f_wt[j]|# disruption around the residue
    global_absdelta= sum_{residues} |f_mut[j] - f_wt[j]|# the per-protein disruption vector

``global_absdelta`` is "the disruption vector" whose magnitude and concentration H1 is
about. All three feed the Phase-3 classifier.
"""

from __future__ import annotations

import numpy as np
import torch

from esm.models.esmc import ESMC

from ..esmc.extract import hidden_states, sae_layer_index
from ..esmc.sae import SAE

_EPS = 1e-6


def mutate(sequence: str, position: int, wt_aa: str, mut_aa: str) -> str:
    """Apply a 1-indexed missense substitution, asserting the WT residue matches."""
    if sequence[position - 1] != wt_aa:
        raise ValueError(
            f"WT mismatch at {position}: seq has {sequence[position-1]}, expected {wt_aa}"
        )
    return sequence[: position - 1] + mut_aa + sequence[position:]


@torch.no_grad()
def sae_residue_features(model: ESMC, sae: SAE, sequence: str) -> torch.Tensor:
    """Per-token SAE features ``[L+2, codebook]`` (incl. <cls>/<eos>) at the SAE layer."""
    hs = hidden_states(model, sequence)[sae_layer_index(sae.layer)]
    return sae.encode(hs)


def disruption_vectors(
    wt_feats: torch.Tensor,
    mut_feats: torch.Tensor,
    position: int,
    window: int = 2,
) -> dict[str, np.ndarray]:
    """Local / window / global disruption vectors (numpy) for one variant."""
    L = wt_feats.shape[0]
    delta = mut_feats - wt_feats  # [L+2, codebook]; token indices align (point mutation)

    local = delta[position]
    lo, hi = max(1, position - window), min(L - 2, position + window)  # residues only
    window_abs = delta[lo : hi + 1].abs().sum(0)
    global_abs = delta[1 : L - 1].abs().sum(0)  # exclude <cls>/<eos>

    return {
        "local_delta": local.float().cpu().numpy(),
        "window_absdelta": window_abs.float().cpu().numpy(),
        "global_absdelta": global_abs.float().cpu().numpy(),
    }


def disruption_scalars(vecs: dict[str, np.ndarray]) -> dict[str, float]:
    """H1 summaries: magnitude + concentration of the per-protein disruption vector."""
    g = vecs["global_absdelta"]
    local = vecs["local_delta"]
    total = float(g.sum())
    active = g[g > _EPS]
    n_active = int(active.size)

    # Normalized Shannon entropy of the disruption distribution: 0 = all mass in one
    # feature (maximally concentrated), 1 = uniform across active features.
    if n_active > 1:
        p = active / active.sum()
        entropy = float(-(p * np.log(p)).sum() / np.log(n_active))
    else:
        entropy = 0.0
    top10_frac = float(np.sort(active)[::-1][:10].sum() / active.sum()) if n_active else 0.0

    return {
        "disruption_total": total,
        "disruption_l2": float(np.linalg.norm(g)),
        "local_l1": float(np.abs(local).sum()),
        "n_features_changed": n_active,
        "entropy": entropy,           # lower = more concentrated
        "top10_frac": top10_frac,     # higher = more concentrated
    }
