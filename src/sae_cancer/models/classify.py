"""Pathogenic-vs-benign classification on disruption vectors, vs baselines.

Headline split is leave-one-gene-out (LOGO): train on some genes, test on an unseen one
(brief §6) — the real generalization test that avoids memorizing per-gene quirks. A
pooled stratified CV (held-out *variants*, mixing genes) is reported as a secondary,
higher-power but more leakage-prone reference.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold

from ..eval.metrics import discrimination

SEED = 0


def make_xgb(y_train: np.ndarray) -> xgb.XGBClassifier:
    n_pos, n_neg = int((y_train == 1).sum()), int((y_train == 0).sum())
    return xgb.XGBClassifier(
        n_estimators=400, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.5, min_child_weight=2,
        tree_method="hist", eval_metric="logloss",
        scale_pos_weight=(n_neg / max(n_pos, 1)), random_state=SEED, n_jobs=-1,
    )


def _rows(X, idx):
    return X[idx] if sp.issparse(X) else X[idx]


def logo_xgb(X, y: np.ndarray, genes: np.ndarray, min_each: int = 10) -> dict:
    """Leave-one-gene-out XGBoost. Test folds = genes with >= min_each of each class."""
    qualifying = [g for g in np.unique(genes)
                  if (y[genes == g] == 1).sum() >= min_each
                  and (y[genes == g] == 0).sum() >= min_each]
    per_gene, oof_y, oof_p = {}, [], []
    for g in qualifying:
        te = genes == g
        tr = ~te
        model = make_xgb(y[tr])
        model.fit(_rows(X, tr), y[tr])
        p = model.predict_proba(_rows(X, te))[:, 1]
        per_gene[g] = discrimination(y[te], p)
        oof_y.append(y[te]); oof_p.append(p)
    pooled = discrimination(np.concatenate(oof_y), np.concatenate(oof_p)) if oof_y else {}
    mean_auroc = float(np.mean([m["auroc"] for m in per_gene.values()])) if per_gene else float("nan")
    return {"test_genes": qualifying, "per_gene": per_gene,
            "mean_auroc": mean_auroc, "pooled": pooled}


def logo_likelihood(score: np.ndarray, y: np.ndarray, genes: np.ndarray, min_each: int = 10) -> dict:
    """Parameter-free baseline on the same qualifying test genes (score: higher=pathogenic)."""
    qualifying = [g for g in np.unique(genes)
                  if (y[genes == g] == 1).sum() >= min_each
                  and (y[genes == g] == 0).sum() >= min_each]
    per_gene = {g: discrimination(y[genes == g], score[genes == g]) for g in qualifying}
    oof_y = np.concatenate([y[genes == g] for g in qualifying]) if qualifying else np.array([])
    oof_s = np.concatenate([score[genes == g] for g in qualifying]) if qualifying else np.array([])
    return {"test_genes": qualifying, "per_gene": per_gene,
            "mean_auroc": float(np.mean([m["auroc"] for m in per_gene.values()])) if per_gene else float("nan"),
            "pooled": discrimination(oof_y, oof_s) if len(oof_y) else {}}


def strat_xgb(X, y: np.ndarray, n_splits: int = 5) -> dict:
    """Pooled stratified-CV out-of-fold AUROC (held-out variants; mixes genes)."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y))
    for tr, te in skf.split(np.zeros(len(y)), y):
        model = make_xgb(y[tr])
        model.fit(_rows(X, tr), y[tr])
        oof[te] = model.predict_proba(_rows(X, te))[:, 1]
    return discrimination(y, oof)
