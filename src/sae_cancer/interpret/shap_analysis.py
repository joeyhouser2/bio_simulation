"""SHAP attribution over the SAE disruption features (brief §5.1.5, §4.4).

Identifies which interpretable SAE features drive pathogenic-vs-benign calls. The
disruption matrix stacks three representations (local / window / global) of the 16,384
codebook, so column ``j`` maps to ``(representation = j // 16384, feature_id = j % 16384)``.
``feature_id`` is the codebook index whose GPT-5 description we cross-reference (H2).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
import shap

from ..models.classify import make_xgb

CODEBOOK = 16384
REPRS = ("local_delta", "window_absdelta", "global_absdelta")


def shap_feature_importance(X_sae: sp.csr_matrix, y: np.ndarray) -> pd.DataFrame:
    """Train XGBoost on all labeled data and rank disruption columns by mean |SHAP|.

    Returns a per-column table: representation, feature_id, mean_abs_shap, plus the
    signed mean SHAP (direction: positive => pushes toward pathogenic).
    """
    model = make_xgb(y)
    model.fit(X_sae, y)

    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_sae)  # [n, n_features]
    mean_abs = np.abs(sv).mean(axis=0)
    mean_signed = sv.mean(axis=0)

    n_cols = X_sae.shape[1]
    cols = np.arange(n_cols)
    df = pd.DataFrame({
        "column": cols,
        "representation": [REPRS[c // CODEBOOK] for c in cols],
        "feature_id": cols % CODEBOOK,
        "mean_abs_shap": mean_abs,
        "mean_signed_shap": mean_signed,
    })
    return df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def aggregate_by_feature(shap_df: pd.DataFrame, top: int = 30) -> pd.DataFrame:
    """Collapse the three representations to per-codebook-feature importance."""
    agg = (shap_df.groupby("feature_id")["mean_abs_shap"].sum()
           .sort_values(ascending=False).head(top).reset_index())
    return agg
