"""The cancer-gene panel and wild-type sequence retrieval.

Phase 1 uses a 5-gene smoke panel (brief §11): three oncogenes and two tumor
suppressors. WT canonical sequences come from UniProt (public, no auth) and are cached
under ``data/raw/sequences/``.
"""

from __future__ import annotations

import json
from pathlib import Path

import requests

# Repo paths.
ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = ROOT / "data" / "raw"
SEQ_CACHE = RAW_DIR / "sequences"

# 5-gene smoke panel: UniProt canonical accession + oncogene/TSG mechanism (H2 label).
PANEL: dict[str, dict[str, str]] = {
    "KRAS": {"uniprot": "P01116", "role": "oncogene"},
    "TP53": {"uniprot": "P04637", "role": "TSG"},
    "BRAF": {"uniprot": "P15056", "role": "oncogene"},
    "PTEN": {"uniprot": "P60484", "role": "TSG"},
    "EGFR": {"uniprot": "P00533", "role": "oncogene"},
}

# UniProt canonical for KRAS is the 4B isoform (P01116-1); hotspot/ClinVar numbering
# (G12, Q61, ...) matches this sequence.

UNIPROT_FASTA = "https://rest.uniprot.org/uniprotkb/{acc}.fasta"


def fetch_sequence(uniprot: str, refresh: bool = False) -> str:
    """Return the canonical UniProt sequence for an accession, cached locally."""
    SEQ_CACHE.mkdir(parents=True, exist_ok=True)
    cache = SEQ_CACHE / f"{uniprot}.json"
    if cache.exists() and not refresh:
        return json.loads(cache.read_text())["sequence"]

    resp = requests.get(UNIPROT_FASTA.format(acc=uniprot), timeout=60)
    resp.raise_for_status()
    seq = "".join(resp.text.splitlines()[1:])
    if not seq:
        raise RuntimeError(f"empty sequence for {uniprot}")
    cache.write_text(json.dumps({"uniprot": uniprot, "sequence": seq}))
    return seq


def panel_sequences(refresh: bool = False) -> dict[str, str]:
    """Gene -> WT sequence for the whole panel."""
    return {g: fetch_sequence(info["uniprot"], refresh) for g, info in PANEL.items()}
