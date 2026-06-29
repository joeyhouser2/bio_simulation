"""Phase 4: build the expanded cancer-gene panel (oncogene/TSG) -> data/variants/panel.csv.

Run:
    .venv\\Scripts\\python.exe scripts\\phase4_build_panel.py [--per-role 40] [--max-len 1500]
"""

from __future__ import annotations

import argparse

from sae_cancer.variants.cancer_genes import save_panel, select_panel


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-role", type=int, default=40)
    ap.add_argument("--max-len", type=int, default=1500)
    args = ap.parse_args()

    df = select_panel(n_per_role=args.per_role, max_len=args.max_len)
    save_panel(df)

    print(f"Selected {len(df)} genes -> data/variants/panel.csv")
    for role, g in df.groupby("role"):
        print(f"  {role:9s} n={len(g):3d}  median_len={int(g['length'].median())}")
        print("    " + " ".join(g["gene"].tolist()))


if __name__ == "__main__":
    main()
