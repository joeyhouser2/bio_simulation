"""Phase 3 / H2: interpret which SAE features cancer mutations disrupt (6B only).

Cross-references the SHAP-ranked disruption features against the GPT-5 feature
descriptions, tests whether disrupted features are biologically coherent (category
enrichment), and whether oncogene vs tumor-suppressor variants disrupt different feature
categories (H2) via a gene-held-out role classification.

Run (after the 6B disruption + SHAP scripts):
    set PYTHONIOENCODING=utf-8
    .venv\\Scripts\\python.exe scripts\\phase3_h2_interpret.py --model esmc_6b
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import scipy.sparse as sp

from sae_cancer.disruption.dataset import load_disruption
from sae_cancer.interpret.descriptions import load_descriptions
from sae_cancer.interpret.mechanism import (
    category_profiles, feature_category_onehot, shap_category_enrichment,
)
from sae_cancer.models.classify import make_xgb
from sae_cancer.eval.metrics import discrimination
from sae_cancer.variants.curate import ROOT

RESULTS = ROOT / "results"


def role_logo_auroc(profiles: np.ndarray, role_onco: np.ndarray, genes: np.ndarray) -> float:
    """Pooled gene-held-out AUROC for oncogene-vs-TSG from disruption category profiles."""
    oof = np.full(len(role_onco), np.nan)
    for g in np.unique(genes):
        te = genes == g
        tr = ~te
        if len(np.unique(role_onco[tr])) < 2:
            continue
        model = make_xgb(role_onco[tr].astype(int))
        model.fit(profiles[tr], role_onco[tr].astype(int))
        oof[te] = model.predict_proba(profiles[te])[:, 1]
    ok = ~np.isnan(oof)
    return discrimination(role_onco[ok].astype(int), oof[ok])["auroc"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="esmc_6b")
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args()

    scalars, mats = load_disruption(args.model)
    desc = load_descriptions().reset_index()  # feature_id as column
    shap_top = pd.read_csv(RESULTS / f"phase3_shap_top_features_{args.model}.csv")
    desc_by_id = desc.set_index("feature_id")

    # --- A. Headline: top SHAP features -> biology -----------------------------------
    print(f"=== Top {args.top} disruption features driving pathogenicity (6B) ===")
    rows = []
    for _, r in shap_top.head(args.top).iterrows():
        fid = int(r["feature_id"])
        d = desc_by_id.loc[fid] if fid in desc_by_id.index else None
        cat = d["category"] if d is not None else "?"
        summ = (d["summary"] if d is not None else "")[:88]
        rows.append({"feature_id": fid, "importance": r["mean_abs_shap"],
                     "category": cat, "summary": d["summary"] if d is not None else ""})
        print(f"  {fid:5d} [{r['mean_abs_shap']:.3f}] {cat:22s} {summ}")
    pd.DataFrame(rows).to_csv(RESULTS / f"phase3_h2_top_features_{args.model}.csv", index=False)

    # --- B. Category enrichment of the predictive features ---------------------------
    enr = shap_category_enrichment(shap_top["feature_id"].tolist(), desc, top_k=50)
    print("\n=== Category enrichment of top-50 SHAP features (obs/background) ===")
    print(enr.round(3).to_string())

    # --- C. Oncogene vs TSG mechanism separation (H2) --------------------------------
    cats, onehot = feature_category_onehot(desc)
    prof = category_profiles(mats["global_absdelta"], onehot)  # [N, n_cat]
    prof = prof / (prof.sum(axis=1, keepdims=True) + 1e-9)     # normalize per variant

    lab = scalars.dropna(subset=["path_label"]).copy()
    mask = scalars["path_label"].notna().values
    patho = mask & (scalars["path_label"].values == 1)
    role_onco = (scalars["role"].values == "oncogene")
    genes = scalars["gene"].values

    # On pathogenic variants only: which categories differ by role (mean normalized).
    pp = prof[patho]
    onco = pp[role_onco[patho]].mean(0)
    tsg = pp[~role_onco[patho]].mean(0)
    diff = pd.DataFrame({"category": cats, "oncogene_mean": onco, "tsg_mean": tsg})
    diff["onco_minus_tsg"] = diff["oncogene_mean"] - diff["tsg_mean"]
    print("\n=== Mean category-disruption profile, pathogenic variants (onco vs TSG) ===")
    print(diff.sort_values("onco_minus_tsg", ascending=False).round(4).to_string(index=False))

    auroc_role = role_logo_auroc(prof[patho], role_onco[patho], genes[patho])
    print(f"\nGene-held-out oncogene-vs-TSG AUROC from category profiles: {auroc_role:.3f}")
    print("  (5-gene panel: preliminary; Phase 4 scale-up needed for a powered claim)")

    report = {
        "model": args.model,
        "category_enrichment": enr.round(4).to_dict(orient="index"),
        "onco_vs_tsg_profile": diff.round(5).to_dict(orient="records"),
        "role_logo_auroc": float(auroc_role),
    }
    (RESULTS / f"phase3_h2_{args.model}.json").write_text(json.dumps(report, indent=2, default=float))
    print(f"\nWrote results/phase3_h2_{args.model}.json and top-features CSV")


if __name__ == "__main__":
    main()
