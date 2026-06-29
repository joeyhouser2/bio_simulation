"""Phase 3: classify pathogenic-vs-benign on disruption vectors, vs both baselines.

Models, all on leakage-proof leave-one-gene-out (LOGO) splits (+ stratified CV ref):
  1. likelihood       - raw masked-marginal LLR (parameter-free spine)
  2. embedding-delta  - XGBoost on dense hidden-state deltas (no SAE)
  3. SAE-disruption   - XGBoost on the sparse feature-disruption vectors
  4. combined         - SAE-disruption + LLR

Run (builds embedding deltas on first call):
    .venv\\Scripts\\python.exe scripts\\phase3_classify.py [--model esmc_600m]
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import scipy.sparse as sp

from sae_cancer.baselines.embedding import build_embedding_deltas, load_embedding_deltas, save_embedding_deltas
from sae_cancer.disruption.dataset import load_disruption
from sae_cancer.esmc.extract import MODEL_REGISTRY
from sae_cancer.models.classify import logo_likelihood, logo_xgb, strat_xgb
from sae_cancer.variants.curate import ROOT
from sae_cancer.variants.panel import panel_sequences

RESULTS = ROOT / "results"
KEYS = ["gene", "position", "wt_aa", "mut_aa"]


def _print_model(name: str, res: dict, strat: dict | None) -> None:
    pg = "  ".join(f"{g}={m['auroc']:.3f}" for g, m in res["per_gene"].items())
    pooled = res["pooled"].get("auroc", float("nan"))
    strat_s = f"  strat-CV={strat['auroc']:.3f}" if strat else ""
    print(f"  {name:16s} LOGO mean={res['mean_auroc']:.3f}  pooled={pooled:.3f}{strat_s}")
    print(f"  {'':16s}   per-gene: {pg}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="esmc_600m")
    args = ap.parse_args()
    mdl = args.model

    scalars, mats = load_disruption(mdl)
    print(f"Disruption: {mats['local_delta'].shape}, {len(scalars)} variants")

    # Embedding-delta baseline (build + cache on first run; same row order as disruption).
    try:
        emb = load_embedding_deltas(mdl)
    except FileNotFoundError:
        from sae_cancer.esmc.extract import load_esmc
        print("Building embedding-delta baseline ...")
        model = load_esmc(mdl)
        df = pd.read_csv(ROOT / "data" / "variants" / "variants.csv")
        emb = build_embedding_deltas(model, df, panel_sequences(), MODEL_REGISTRY[mdl]["sae_layer"])
        save_embedding_deltas(mdl, emb)
    print(f"Embedding deltas: {emb['emb_local_delta'].shape}")

    # Align likelihood scores to disruption row order.
    llr = pd.read_csv(RESULTS / f"phase1_likelihood_{mdl}.csv")[KEYS + ["llr"]]
    merged = scalars.merge(llr, on=KEYS, how="left")
    assert len(merged) == len(scalars)

    # Labeled rows only.
    mask = merged["path_label"].notna().values & merged["llr"].notna().values
    rows = np.where(mask)[0]
    y = merged.loc[mask, "path_label"].astype(int).values
    genes = merged.loc[mask, "gene"].values
    llr_score = -merged.loc[mask, "llr"].values  # higher = more pathogenic

    # Feature matrices.
    X_sae = sp.hstack([mats[r][rows] for r in ("local_delta", "window_absdelta", "global_absdelta")]).tocsr()
    X_emb = np.hstack([emb["emb_local_delta"][rows], emb["emb_global_absdelta"][rows]])
    X_comb = sp.hstack([X_sae, sp.csr_matrix(llr_score.reshape(-1, 1))]).tocsr()
    print(f"X_sae={X_sae.shape}  X_emb={X_emb.shape}  n_labeled={len(y)} "
          f"(path={int(y.sum())}, benign={int((y==0).sum())})\n")

    out = {"model": mdl, "n_labeled": int(len(y))}
    print("=== Pathogenic-vs-benign (AUROC) ===")

    out["likelihood"] = logo_likelihood(llr_score, y, genes)
    _print_model("likelihood", out["likelihood"], None)

    out["embedding"] = logo_xgb(X_emb, y, genes)
    _print_model("embedding-delta", out["embedding"], strat_xgb(X_emb, y))

    out["sae_disruption"] = logo_xgb(X_sae, y, genes)
    _print_model("SAE-disruption", out["sae_disruption"], strat_xgb(X_sae, y))

    out["combined"] = logo_xgb(X_comb, y, genes)
    _print_model("combined", out["combined"], strat_xgb(X_comb, y))

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"phase3_classify_{mdl}.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"\nTest genes (>=10 each class): {out['likelihood']['test_genes']}")
    print(f"Wrote results/phase3_classify_{mdl}.json")


if __name__ == "__main__":
    main()
