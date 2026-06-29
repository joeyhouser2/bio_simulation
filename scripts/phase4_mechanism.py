"""Phase 4 / H2 (powered): does SAE feature-disruption separate oncogenes from TSGs?

Classifies oncogene-vs-TSG from the disruption vectors under leave-one-gene-out (pooled
OOF) — the leakage-proof mechanism test the 5-gene panel couldn't support. Works on any
model's cached disruption (no GPT-5 descriptions needed; that's the category-level
interpretation, 6B-only).

Run:
    .venv\\Scripts\\python.exe scripts\\phase4_mechanism.py [--model esmc_600m]
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import scipy.sparse as sp

from sae_cancer.disruption.dataset import load_disruption
from sae_cancer.models.classify import logo_pooled_xgb
from sae_cancer.variants.curate import ROOT

RESULTS = ROOT / "results"
REPRS = ("local_delta", "window_absdelta", "global_absdelta")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="esmc_600m")
    args = ap.parse_args()

    scalars, mats = load_disruption(args.model)
    genes_all = scalars["gene"].values
    role_onco = (scalars["role"].values == "oncogene").astype(int)
    X = sp.hstack([mats[r] for r in REPRS]).tocsr()

    out = {"model": args.model, "n_genes": int(len(np.unique(genes_all)))}
    print(f"Oncogene-vs-TSG from disruption vectors (LOGO pooled), {args.model}")
    print(f"  {len(scalars)} variants across {out['n_genes']} genes "
          f"({int(role_onco.sum())} oncogene-variants, {int((role_onco==0).sum())} TSG)")

    # All variants.
    res_all = logo_pooled_xgb(X, role_onco, genes_all)
    out["all_variants"] = res_all
    print(f"  ALL variants:        AUROC={res_all['auroc']:.3f}  (held-out genes={res_all['n_genes']})")

    # Pathogenic variants only (the functionally disruptive ones).
    patho = scalars["path_label"].fillna(-1).values == 1
    if patho.sum() > 50:
        res_p = logo_pooled_xgb(X[patho], role_onco[patho], genes_all[patho])
        out["pathogenic_only"] = res_p
        print(f"  pathogenic variants: AUROC={res_p['auroc']:.3f}  (held-out genes={res_p['n_genes']})")

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"phase4_mechanism_{args.model}.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"\nWrote results/phase4_mechanism_{args.model}.json")


if __name__ == "__main__":
    main()
