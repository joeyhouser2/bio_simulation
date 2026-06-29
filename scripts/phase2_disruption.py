"""Phase 2: WT->mutant SAE feature-disruption + H1 sanity check.

Builds per-variant disruption vectors (cached under data/features/) and tests H1 on the
smoke panel: do pathogenic / driver mutations cause LARGER and MORE CONCENTRATED
disruption of SAE features than benign / non-hotspot ones?

Run:
    .venv\\Scripts\\python.exe scripts\\phase2_disruption.py [--model esmc_600m]
"""

from __future__ import annotations

import argparse
import json

import numpy as np
from scipy.stats import mannwhitneyu

from sae_cancer.disruption.dataset import build_disruption, load_disruption, save_disruption
from sae_cancer.esmc.extract import load_esmc
from sae_cancer.esmc.sae import load_sae
from sae_cancer.eval.metrics import discrimination
from sae_cancer.variants.curate import ROOT, load_variants
from sae_cancer.variants.panel import panel_sequences

RESULTS = ROOT / "results"
SAE_REPOS = {
    "esmc_300m": "biohub/ESMC-300M-sae-layer23-k64-codebook16384",
    "esmc_600m": "biohub/ESMC-600M-sae-layer27-k64-codebook16384",
    "esmc_6b": "biohub/ESMC-6B-sae-layer60-k64-codebook16384",
}

# H1 scalars and the direction that should indicate pathogenic/driver (sign * scalar
# is the "more pathogenic" predictor). Magnitude up; entropy down; top10_frac up.
H1_SCALARS = {
    "disruption_total": +1, "disruption_l2": +1, "local_l1": +1,
    "n_features_changed": +1, "top10_frac": +1, "entropy": -1,
}


def _gene_stratified_auroc(sub, s, sign, min_each=5) -> tuple[float, int, dict]:
    """Mean per-gene AUROC over genes with >= min_each of each class (confound-controlled)."""
    per_gene, aurocs = {}, []
    for gene, g in sub.groupby("gene"):
        y = g["_y"].values
        if (y == 1).sum() < min_each or (y == 0).sum() < min_each:
            continue
        au = discrimination(y, sign * g[s].values)["auroc"]
        per_gene[gene] = au
        aurocs.append(au)
    mean = float(np.mean(aurocs)) if aurocs else float("nan")
    return mean, len(aurocs), per_gene


def h1_report(scalars, label_col: str, pos_name: str, neg_name: str) -> dict:
    sub = scalars.dropna(subset=[label_col]).copy()
    sub["_y"] = sub[label_col].astype(int)
    y = sub["_y"].values
    out = {"n_pos": int((y == 1).sum()), "n_neg": int((y == 0).sum()), "scalars": {}}
    print(f"\n=== H1: {pos_name} (n={out['n_pos']}) vs {neg_name} (n={out['n_neg']}) ===")
    print(f"  {'scalar':18s} {'pooled':>7} {'gene-strat':>11} {'(#genes)':>9} {'MWU p':>10}")
    print(f"  {'(sign=H1 dir)':18s} {'AUROC':>7} {'AUROC':>11}")
    for s, sign in H1_SCALARS.items():
        pooled = discrimination(y, sign * sub[s].values)["auroc"]
        strat, n_genes, per_gene = _gene_stratified_auroc(sub, s, sign)
        try:
            p = float(mannwhitneyu(sub[s].values[y == 1], sub[s].values[y == 0],
                                   alternative="two-sided").pvalue)
        except ValueError:
            p = float("nan")
        out["scalars"][s] = {"auroc_pooled": pooled, "auroc_gene_stratified": strat,
                             "n_genes": n_genes, "per_gene": per_gene, "mwu_p": p}
        print(f"  {s:18s} {pooled:7.3f} {strat:11.3f} {n_genes:>9d} {p:10.2e}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="esmc_600m")
    ap.add_argument("--rebuild", action="store_true", help="recompute even if cached")
    args = ap.parse_args()

    try:
        if args.rebuild:
            raise FileNotFoundError
        scalars, mats = load_disruption(args.model)
        print(f"Loaded cached disruption for {args.model}: {mats['local_delta'].shape}")
    except FileNotFoundError:
        df = load_variants()
        seqs = panel_sequences()
        print(f"Loading {args.model} + SAE ...")
        model = load_esmc(args.model)
        sae = load_sae(SAE_REPOS[args.model], device=model.device)
        print(f"  model+SAE on {model.device}; scoring {len(df)} variants "
              f"(+{df['gene'].nunique()} WT)")
        scalars, mats = build_disruption(model, sae, df, seqs)
        save_disruption(args.model, scalars, mats)
        print(f"\nCached disruption: scalars + {list(mats)} "
              f"({mats['local_delta'].shape}) -> data/features/")

    report = {"model": args.model}
    report["pathogenic_vs_benign"] = h1_report(scalars, "path_label", "pathogenic", "benign")
    scalars["_hot"] = scalars["is_hotspot"].astype(int)
    report["hotspot_vs_nonhotspot"] = h1_report(scalars, "_hot", "hotspot", "non-hotspot")

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"phase2_h1_{args.model}.json").write_text(json.dumps(report, indent=2))
    print(f"\nWrote results/phase2_h1_{args.model}.json")


if __name__ == "__main__":
    main()
