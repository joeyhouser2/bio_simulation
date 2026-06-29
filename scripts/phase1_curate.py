"""Phase 1: curate the labeled variant table for the 5-gene smoke panel.

Pulls ClinVar (pathogenic/benign missense) + Cancer Hotspots (drivers), validates
against WT UniProt sequences, and writes data/variants/variants.csv.

Run:
    .venv\\Scripts\\python.exe scripts\\phase1_curate.py
"""

from __future__ import annotations

import argparse

from sae_cancer.variants.curate import build_variant_table, save_variant_table


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="re-download sources")
    args = ap.parse_args()

    print("Building variant table (ClinVar + Cancer Hotspots) ...")
    df = build_variant_table(refresh=args.refresh)
    path = save_variant_table(df)

    stats = df.attrs.get("stats", {})
    print(f"\nFetched: clinvar={stats.get('clinvar')} hotspot={stats.get('hotspot')} "
          f"dropped(seq mismatch)={stats.get('dropped_seq_mismatch')}")
    print(f"Wrote {len(df)} validated variants -> {path}\n")

    print("Per-gene summary:")
    for gene, g in df.groupby("gene"):
        path_lab = g["path_label"].dropna()
        n_path = int((path_lab == 1).sum())
        n_benign = int((path_lab == 0).sum())
        n_hot = int(g["is_hotspot"].sum())
        print(f"  {gene:5s} role={g['role'].iloc[0]:9s} n={len(g):4d}  "
              f"pathogenic={n_path:4d} benign={n_benign:4d} hotspot={n_hot:4d}")

    print("\npathogenic-vs-benign label balance (all genes):")
    pl = df["path_label"].dropna()
    print(f"  pathogenic={int((pl==1).sum())}  benign={int((pl==0).sum())}  "
          f"unlabeled(hotspot-only)={int(df['path_label'].isna().sum())}")


if __name__ == "__main__":
    main()
