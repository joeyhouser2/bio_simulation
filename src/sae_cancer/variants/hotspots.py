"""Recurrent somatic driver mutations from Cancer Hotspots (cancerhotspots.org).

Public JSON API, no auth. We expand each single-residue hotspot into its observed
amino-acid substitutions (the driver signal for the driver-vs-passenger task and an
oncogenicity tag for the pathogenic-vs-benign task). Cached under ``data/raw/``.
"""

from __future__ import annotations

import json
from pathlib import Path

import requests

from .panel import PANEL, RAW_DIR

HOTSPOTS_API = "https://www.cancerhotspots.org/api/hotspots/single"
CACHE = RAW_DIR / "cancerhotspots_single.json"


def _load_raw(refresh: bool = False) -> list[dict]:
    if CACHE.exists() and not refresh:
        return json.loads(CACHE.read_text())
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(HOTSPOTS_API, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    CACHE.write_text(json.dumps(data))
    return data


def fetch_hotspot_variants(refresh: bool = False) -> list[dict]:
    """Return per-substitution hotspot rows for the panel genes.

    Each row: ``gene, position, wt_aa, mut_aa, tumor_count, hotspot_qvalue``.
    ``wt_aa`` comes from the hotspot residue label (e.g. "R175" -> R); rows whose
    variant amino acid is a stop/splice marker are skipped.
    """
    raw = _load_raw(refresh)
    genes = set(PANEL)
    rows: list[dict] = []
    for h in raw:
        gene = h.get("hugoSymbol")
        if gene not in genes:
            continue
        residue = h.get("residue", "")  # e.g. "R175" or "Q61"
        wt_aa = residue[0] if residue and residue[0].isalpha() else None
        pos = h.get("aminoAcidPosition", {}).get("start")
        if pos is None:
            continue
        for mut_aa, count in (h.get("variantAminoAcid") or {}).items():
            if len(mut_aa) != 1 or not mut_aa.isalpha() or mut_aa == "*":
                continue
            rows.append({
                "gene": gene, "position": int(pos), "wt_aa": wt_aa, "mut_aa": mut_aa,
                "tumor_count": int(count), "hotspot_qvalue": float(h.get("qValue", 1)),
            })
    return rows
