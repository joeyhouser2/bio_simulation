"""Phase 3: SHAP attribution over the SAE disruption features.

Ranks which interpretable codebook features drive pathogenic-vs-benign calls and writes
a table of top features (by codebook feature_id) for GPT-5-description cross-reference.

Run:
    .venv\\Scripts\\python.exe scripts\\phase3_shap.py [--model esmc_600m] [--top 40]
"""

from __future__ import annotations

import argparse

import numpy as np
import scipy.sparse as sp

from sae_cancer.disruption.dataset import load_disruption
from sae_cancer.interpret.shap_analysis import aggregate_by_feature, shap_feature_importance
from sae_cancer.variants.curate import ROOT

RESULTS = ROOT / "results"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="esmc_600m")
    ap.add_argument("--top", type=int, default=40)
    args = ap.parse_args()

    scalars, mats = load_disruption(args.model)
    mask = scalars["path_label"].notna().values
    rows = np.where(mask)[0]
    y = scalars.loc[mask, "path_label"].astype(int).values
    X_sae = sp.hstack(
        [mats[r][rows] for r in ("local_delta", "window_absdelta", "global_absdelta")]
    ).tocsr()
    print(f"SHAP on {X_sae.shape}, path={int(y.sum())} benign={int((y==0).sum())}")

    shap_df = shap_feature_importance(X_sae, y)
    by_feat = aggregate_by_feature(shap_df, top=args.top)

    RESULTS.mkdir(exist_ok=True)
    shap_df.head(200).to_csv(RESULTS / f"phase3_shap_columns_{args.model}.csv", index=False)
    by_feat.to_csv(RESULTS / f"phase3_shap_top_features_{args.model}.csv", index=False)

    print(f"\nTop {args.top} codebook features by summed mean|SHAP| (for GPT-5 cross-ref):")
    print(f"  {'feature_id':>10} {'importance':>12}")
    for _, r in by_feat.iterrows():
        print(f"  {int(r['feature_id']):>10} {r['mean_abs_shap']:>12.4f}")
    print(f"\nWrote results/phase3_shap_top_features_{args.model}.csv")


if __name__ == "__main__":
    main()
