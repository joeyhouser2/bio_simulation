"""Assemble the labeled variant table for the panel.

Merges ClinVar germline missense (pathogenic/benign) with Cancer Hotspots recurrent
somatic drivers, validates every variant against the WT UniProt sequence, and writes a
committed table at ``data/variants/variants.csv``. This is the reproducible input to the
likelihood baseline and the SAE feature pipeline.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .clinvar import fetch_clinvar_missense
from .hotspots import fetch_hotspot_variants
from .panel import PANEL, ROOT, panel_sequences

OUT_DIR = ROOT / "data" / "variants"
OUT_CSV = OUT_DIR / "variants.csv"

COLUMNS = [
    "gene", "uniprot", "role", "position", "wt_aa", "mut_aa", "hgvs_p",
    "path_label", "clinsig", "is_hotspot", "tumor_count",
]


def build_variant_table(refresh: bool = False) -> pd.DataFrame:
    seqs = panel_sequences(refresh)
    merged: dict[tuple, dict] = {}
    stats: dict[str, int] = {"clinvar": 0, "hotspot": 0, "dropped_seq_mismatch": 0}

    def key(gene, pos, wt, mut):
        return (gene, pos, wt, mut)

    # ClinVar pathogenic/benign.
    for gene in PANEL:
        for r in fetch_clinvar_missense(gene, refresh):
            merged[key(gene, r["position"], r["wt_aa"], r["mut_aa"])] = {
                "gene": gene, "uniprot": PANEL[gene]["uniprot"], "role": PANEL[gene]["role"],
                "position": r["position"], "wt_aa": r["wt_aa"], "mut_aa": r["mut_aa"],
                "hgvs_p": r["hgvs_p"], "path_label": r["path_label"], "clinsig": r["clinsig"],
                "is_hotspot": False, "tumor_count": 0,
            }
            stats["clinvar"] += 1

    # Cancer Hotspots drivers (merge onto existing rows; add new ones).
    for r in fetch_hotspot_variants(refresh):
        k = key(r["gene"], r["position"], r["wt_aa"], r["mut_aa"])
        if k in merged:
            merged[k]["is_hotspot"] = True
            merged[k]["tumor_count"] = r["tumor_count"]
        else:
            g = r["gene"]
            wt3 = r["wt_aa"] or "?"
            merged[k] = {
                "gene": g, "uniprot": PANEL[g]["uniprot"], "role": PANEL[g]["role"],
                "position": r["position"], "wt_aa": r["wt_aa"], "mut_aa": r["mut_aa"],
                "hgvs_p": f"p.{wt3}{r['position']}{r['mut_aa']}",
                "path_label": pd.NA, "clinsig": pd.NA,
                "is_hotspot": True, "tumor_count": r["tumor_count"],
            }
        stats["hotspot"] += 1

    # Validate against WT sequence: wt_aa must match seq[pos-1].
    rows = []
    for v in merged.values():
        seq = seqs[v["gene"]]
        pos = v["position"]
        if v["wt_aa"] is None or pos < 1 or pos > len(seq) or seq[pos - 1] != v["wt_aa"]:
            stats["dropped_seq_mismatch"] += 1
            continue
        rows.append(v)

    df = pd.DataFrame(rows, columns=COLUMNS).sort_values(
        ["gene", "position", "mut_aa"]
    ).reset_index(drop=True)
    df.attrs["stats"] = stats
    return df


def save_variant_table(df: pd.DataFrame) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    return OUT_CSV


def load_variants() -> pd.DataFrame:
    return pd.read_csv(OUT_CSV)
