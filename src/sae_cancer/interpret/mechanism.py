"""H2 mechanism analysis: do disrupted SAE features cohere biologically, and do they
differ between oncogenes and tumor suppressors? (brief H2 / §4.4)

Builds per-variant *category* disruption profiles (collapsing the 16,384 features to the
14 GPT-5 categories) and tools to test category enrichment of the SHAP-ranked features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

CODEBOOK = 16384


def feature_category_onehot(descriptions: pd.DataFrame) -> tuple[list[str], sp.csr_matrix]:
    """One-hot ``[CODEBOOK, n_categories]`` mapping each feature_id to its category."""
    cats = sorted(descriptions["category"].dropna().unique())
    cat_idx = {c: i for i, c in enumerate(cats)}
    by_id = descriptions.set_index("feature_id")["category"]
    rows, cols = [], []
    for fid in range(CODEBOOK):
        c = by_id.get(fid)
        if isinstance(c, str) and c in cat_idx:
            rows.append(fid)
            cols.append(cat_idx[c])
    data = np.ones(len(rows))
    onehot = sp.csr_matrix((data, (rows, cols)), shape=(CODEBOOK, len(cats)))
    return cats, onehot


def category_profiles(global_absdelta: sp.csr_matrix, onehot: sp.csr_matrix) -> np.ndarray:
    """Per-variant disruption summed within each category: ``[N, n_categories]``."""
    return np.asarray((global_absdelta @ onehot).todense())


def shap_category_enrichment(
    top_feature_ids: list[int], descriptions: pd.DataFrame, top_k: int | None = None
) -> pd.DataFrame:
    """Category composition of the top SHAP features vs the codebook background.

    enrichment = observed_fraction / background_fraction (>1 = over-represented).
    """
    if top_k:
        top_feature_ids = top_feature_ids[:top_k]
    by_id = descriptions.set_index("feature_id")["category"]
    bg = descriptions["category"].value_counts(normalize=True)
    obs = by_id.loc[[f for f in top_feature_ids if f in by_id.index]].value_counts(normalize=True)
    out = pd.DataFrame({"top_frac": obs, "background_frac": bg}).fillna(0.0)
    out["enrichment"] = out["top_frac"] / out["background_frac"].replace(0, np.nan)
    return out.sort_values("enrichment", ascending=False)
