"""Build, cache, and reload the panel's disruption feature matrices.

Caches the expensive SAE artifact (brief §10): per-variant disruption vectors are
stored as sparse matrices under ``data/features/`` keyed by model name, alongside a
scalar table for the H1 analysis. WT features are computed once per gene.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from esm.models.esmc import ESMC

from ..esmc.sae import SAE
from ..variants.panel import ROOT
from .compute import disruption_scalars, disruption_vectors, mutate, sae_residue_features

FEATURES_DIR = ROOT / "data" / "features"
REPRS = ("local_delta", "window_absdelta", "global_absdelta")


def build_disruption(
    model: ESMC, sae: SAE, df: pd.DataFrame, seqs: dict[str, str], window: int = 2,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict[str, sp.csr_matrix]]:
    """Compute disruption vectors + scalars for every variant in ``df`` (row order kept)."""
    rows: list[dict] = []
    vecs: dict[str, list[np.ndarray]] = {r: [] for r in REPRS}

    t0 = time.time()
    for gene, g in df.groupby("gene", sort=False):
        wt_feats = sae_residue_features(model, sae, seqs[gene])  # once per gene
        for _, v in g.iterrows():
            mut_seq = mutate(seqs[gene], int(v["position"]), v["wt_aa"], v["mut_aa"])
            mut_feats = sae_residue_features(model, sae, mut_seq)
            d = disruption_vectors(wt_feats, mut_feats, int(v["position"]), window)
            for r in REPRS:
                vecs[r].append(d[r])
            rows.append({**v.to_dict(), **disruption_scalars(d)})
        if verbose:
            print(f"  {gene:5s}: {len(g)} variants  ({time.time()-t0:.1f}s elapsed)")

    scalars = pd.DataFrame(rows)
    mats = {r: sp.csr_matrix(np.vstack(vecs[r])) for r in REPRS}
    return scalars, mats


def save_disruption(model_name: str, scalars: pd.DataFrame, mats: dict[str, sp.csr_matrix]) -> Path:
    # CSV (not parquet): the disruption pipeline has torch loaded, and importing pyarrow
    # after torch crashes on Windows. Sparse vectors go to .npz (numpy, no pyarrow).
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    scalars.to_csv(FEATURES_DIR / f"disruption_scalars_{model_name}.csv", index=False)
    for r, m in mats.items():
        sp.save_npz(FEATURES_DIR / f"disruption_{r}_{model_name}.npz", m)
    return FEATURES_DIR


def load_disruption(model_name: str) -> tuple[pd.DataFrame, dict[str, sp.csr_matrix]]:
    scalars = pd.read_csv(FEATURES_DIR / f"disruption_scalars_{model_name}.csv")
    mats = {r: sp.load_npz(FEATURES_DIR / f"disruption_{r}_{model_name}.npz") for r in REPRS}
    return scalars, mats
