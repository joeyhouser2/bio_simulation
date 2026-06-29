"""Embedding-delta baseline (brief §5.2): plain dense ESMC embedding deltas, no SAE.

Isolates what the sparse/interpretable SAE decomposition adds over the dense
representation. Computed at the *same* layer the SAE uses, so the only difference vs the
disruption features is sparse-codebook vs dense-hidden-state. Mirrors the disruption
build's row order (groupby gene, sort=False) so rows align with the cached SAE matrices.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from esm.models.esmc import ESMC

from ..esmc.extract import hidden_states, sae_layer_index
from ..disruption.compute import mutate
from ..disruption.dataset import FEATURES_DIR


@torch.no_grad()
def build_embedding_deltas(
    model: ESMC, df: pd.DataFrame, seqs: dict[str, str], sae_layer: int,
    window: int = 2, verbose: bool = True,
) -> dict[str, np.ndarray]:
    """Dense hidden-state deltas per variant: local (at residue) and global (per-protein)."""
    idx = sae_layer_index(sae_layer)
    local, global_abs = [], []
    t0 = time.time()
    for gene, g in df.groupby("gene", sort=False):
        h_wt = hidden_states(model, seqs[gene])[idx]  # [L+2, d_model]
        for _, v in g.iterrows():
            pos = int(v["position"])
            mut_seq = mutate(seqs[gene], pos, v["wt_aa"], v["mut_aa"])
            h_mut = hidden_states(model, mut_seq)[idx]
            d = (h_mut - h_wt).float()
            local.append(d[pos].cpu().numpy())
            global_abs.append(d[1:-1].abs().sum(0).cpu().numpy())
        if verbose:
            print(f"  {gene:5s}: {len(g)} variants  ({time.time()-t0:.1f}s)")
    return {"emb_local_delta": np.vstack(local), "emb_global_absdelta": np.vstack(global_abs)}


def save_embedding_deltas(model_name: str, arrs: dict[str, np.ndarray]) -> Path:
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(FEATURES_DIR / f"embedding_deltas_{model_name}.npz", **arrs)
    return FEATURES_DIR


def load_embedding_deltas(model_name: str) -> dict[str, np.ndarray]:
    z = np.load(FEATURES_DIR / f"embedding_deltas_{model_name}.npz")
    return {k: z[k] for k in z.files}
