"""Discrimination metrics for variant-effect scores."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def discrimination(y_true: np.ndarray, score: np.ndarray) -> dict[str, float]:
    """AUROC/AUPRC for a higher-score-means-positive predictor.

    Returns NaN metrics (not an exception) when only one class is present, so
    per-gene tables with degenerate label balance still render.
    """
    y_true = np.asarray(y_true)
    score = np.asarray(score)
    mask = ~np.isnan(score)
    y_true, score = y_true[mask], score[mask]
    n_pos, n_neg = int((y_true == 1).sum()), int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return {"auroc": float("nan"), "auprc": float("nan"), "n_pos": n_pos, "n_neg": n_neg}
    return {
        "auroc": float(roc_auc_score(y_true, score)),
        "auprc": float(average_precision_score(y_true, score)),
        "n_pos": n_pos,
        "n_neg": n_neg,
    }
