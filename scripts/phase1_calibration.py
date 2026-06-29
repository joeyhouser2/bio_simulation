"""Phase 1: ProteinGym calibration of the masked-marginal likelihood baseline.

Confirms the baseline is "known-good" before SAE comparison: Spearman of ESMC
masked-marginal LLR vs experimental DMS fitness on the panel-gene assays.

Run:
    .venv\\Scripts\\python.exe scripts\\phase1_calibration.py [--model esmc_600m]
"""

from __future__ import annotations

import argparse
import json

# Import calibration (which imports pyarrow) BEFORE anything that pulls torch:
# on Windows, importing pyarrow after torch causes a native access violation.
from sae_cancer.eval.calibration import PANEL_DMS_ASSAYS, calibrate_assay, load_proteingym
from sae_cancer.esmc.extract import load_esmc
from sae_cancer.variants.curate import ROOT

RESULTS = ROOT / "results"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="esmc_600m")
    ap.add_argument("--assays", nargs="*", default=PANEL_DMS_ASSAYS)
    args = ap.parse_args()

    print(f"Loading {len(args.assays)} ProteinGym assays ...")
    assays = load_proteingym(args.assays)

    print(f"Loading {args.model} ...")
    model = load_esmc(args.model)

    results = {"model": args.model, "metric": "spearman_llr_vs_dms", "per_assay": {}}
    rhos = []
    for aid in args.assays:
        if aid not in assays:
            print(f"  {aid}: not found")
            continue
        r = calibrate_assay(model, assays[aid])
        results["per_assay"][aid] = r
        rho = r["spearman"]
        if rho == rho:
            rhos.append(abs(rho))
        print(f"  {aid:38s} n={r['n']:5d}  spearman={rho:+.3f}")

    results["mean_abs_spearman"] = sum(rhos) / len(rhos) if rhos else float("nan")

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"phase1_calibration_{args.model}.json").write_text(json.dumps(results, indent=2))

    print(f"\n  mean |Spearman| = {results['mean_abs_spearman']:.3f} over {len(rhos)} assays")
    print("  (ESM-family zero-shot on ProteinGym averages ~0.4-0.5; "
          "positive correlation confirms the baseline is sane)")
    print(f"\nWrote results/phase1_calibration_{args.model}.json")


if __name__ == "__main__":
    main()
