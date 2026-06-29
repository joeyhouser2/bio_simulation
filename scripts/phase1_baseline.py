"""Phase 1: masked-marginal likelihood baseline on the curated panel.

Computes the ESMC masked-marginal LLR for every variant, writes scores, and reports
pathogenic-vs-benign discrimination (overall + per gene). This is the spine every later
model is measured against (brief §5.2).

Run:
    .venv\\Scripts\\python.exe scripts\\phase1_baseline.py [--model esmc_600m]
"""

from __future__ import annotations

import argparse
import json
import time

from sae_cancer.baselines.likelihood import masked_marginal_llr
from sae_cancer.esmc.extract import load_esmc
from sae_cancer.variants.curate import ROOT, load_variants
from sae_cancer.variants.panel import panel_sequences
from sae_cancer.eval.metrics import discrimination

RESULTS = ROOT / "results"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="esmc_600m")
    args = ap.parse_args()

    df = load_variants()
    seqs = panel_sequences()
    print(f"Loaded {len(df)} variants across {df['gene'].nunique()} genes")

    print(f"Loading {args.model} ...")
    model = load_esmc(args.model)
    print(f"  on {model.device}")

    df["llr"] = float("nan")
    t0 = time.time()
    for gene, g in df.groupby("gene"):
        variants = list(zip(g["position"], g["wt_aa"], g["mut_aa"]))
        scores = masked_marginal_llr(model, seqs[gene], variants)
        df.loc[g.index, "llr"] = [scores[v] for v in variants]
        print(f"  {gene:5s}: scored {len(g)} variants")
    print(f"Scored all variants in {time.time()-t0:.1f}s")

    RESULTS.mkdir(exist_ok=True)
    out_csv = RESULTS / f"phase1_likelihood_{args.model}.csv"
    df.to_csv(out_csv, index=False)

    # Pathogenic-vs-benign: more negative LLR => pathogenic, so predictor = -llr.
    labeled = df.dropna(subset=["path_label"])
    metrics = {"task": "pathogenic_vs_benign", "model": args.model, "overall": {}, "per_gene": {}}
    metrics["overall"] = discrimination(labeled["path_label"].values, -labeled["llr"].values)
    for gene, g in labeled.groupby("gene"):
        metrics["per_gene"][gene] = discrimination(g["path_label"].values, -g["llr"].values)

    (RESULTS / f"phase1_metrics_{args.model}.json").write_text(json.dumps(metrics, indent=2))

    o = metrics["overall"]
    print("\n=== Pathogenic-vs-benign (masked-marginal likelihood) ===")
    print(f"  OVERALL  AUROC={o['auroc']:.3f}  AUPRC={o['auprc']:.3f}  "
          f"(n_path={o['n_pos']}, n_benign={o['n_neg']})")
    print("  per gene:")
    for gene, m in metrics["per_gene"].items():
        au = f"{m['auroc']:.3f}" if m["auroc"] == m["auroc"] else "  n/a"
        ap_ = f"{m['auprc']:.3f}" if m["auprc"] == m["auprc"] else "  n/a"
        print(f"    {gene:5s} AUROC={au} AUPRC={ap_}  (n_path={m['n_pos']}, n_benign={m['n_neg']})")
    print(f"\nWrote {out_csv.name} and metrics json -> results/")


if __name__ == "__main__":
    main()
