"""Build an expanded cancer-gene panel with oncogene/TSG roles (Phase 4).

Roles come from the public OncoKB cancer-gene list (``geneType``); sequences/lengths from
UniProt. We select prominent, single-role genes (in the Vogelstein or Sanger CGC sets)
balanced across oncogene/TSG, length-capped to keep extraction tractable. The result is
written to ``data/variants/panel.csv`` and picked up by ``panel.py``.
"""

from __future__ import annotations

import json

import pandas as pd
import requests

from .panel import RAW_DIR, ROOT

ONCOKB_URL = "https://www.oncokb.org/api/v1/utils/cancerGeneList"
ONCOKB_CACHE = RAW_DIR / "oncokb_cancer_genes.json"
ACC_CACHE = RAW_DIR / "uniprot_accessions.json"
PANEL_CSV = ROOT / "data" / "variants" / "panel.csv"

ROLE = {"ONCOGENE": "oncogene", "TSG": "TSG"}


def fetch_oncokb(refresh: bool = False) -> list[dict]:
    if ONCOKB_CACHE.exists() and not refresh:
        return json.loads(ONCOKB_CACHE.read_text())
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    data = requests.get(ONCOKB_URL, timeout=120).json()
    ONCOKB_CACHE.write_text(json.dumps(data))
    return data


def _load_acc_cache() -> dict:
    return json.loads(ACC_CACHE.read_text()) if ACC_CACHE.exists() else {}


def uniprot_accession(gene: str, cache: dict) -> tuple[str, int] | None:
    """Reviewed human canonical accession + length for a gene symbol (cached)."""
    if gene in cache:
        v = cache[gene]
        return (v[0], v[1]) if v else None
    q = f"(gene_exact:{gene}) AND (organism_id:9606) AND (reviewed:true)"
    r = requests.get(
        "https://rest.uniprot.org/uniprotkb/search",
        params={"query": q, "fields": "accession,length", "format": "tsv", "size": 1},
        timeout=60,
    )
    lines = r.text.strip().splitlines() if r.ok else []
    result = None
    if len(lines) > 1:
        acc, length = lines[1].split("\t")[:2]
        result = (acc, int(length))
    cache[gene] = list(result) if result else None
    return result


def select_panel(n_per_role: int = 40, max_len: int = 1500) -> pd.DataFrame:
    """Pick prominent single-role genes balanced across oncogene/TSG, length-capped."""
    genes = fetch_oncokb()
    cache = _load_acc_cache()
    rows = []
    for role_key, role in ROLE.items():
        # Prominent = in Vogelstein or Sanger CGC; rank by occurrence.
        cands = [g for g in genes if g.get("geneType") == role_key
                 and (g.get("vogelstein") or g.get("sangerCGC"))]
        cands.sort(key=lambda g: g.get("occurrenceCount", 0), reverse=True)
        kept = 0
        for g in cands:
            if kept >= n_per_role:
                break
            sym = g["hugoSymbol"]
            acc = uniprot_accession(sym, cache)
            if acc is None or acc[1] > max_len:
                continue
            rows.append({"gene": sym, "uniprot": acc[0], "role": role, "length": acc[1]})
            kept += 1
    ACC_CACHE.write_text(json.dumps(cache))
    df = pd.DataFrame(rows).drop_duplicates("gene").sort_values(["role", "gene"])
    return df.reset_index(drop=True)


def save_panel(df: pd.DataFrame) -> None:
    PANEL_CSV.parent.mkdir(parents=True, exist_ok=True)
    df[["gene", "uniprot", "role"]].to_csv(PANEL_CSV, index=False)
