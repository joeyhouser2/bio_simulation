"""ClinVar germline missense labels (pathogenic vs. benign) via NCBI E-utilities.

Public, no API key. We pass a tool name + contact email per NCBI etiquette. Raw
responses are cached under ``data/raw/clinvar/`` so curation is reproducible offline.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import requests

from .panel import RAW_DIR

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
TOOL = "sae-cancer"
EMAIL = "jth156@case.edu"
CACHE = RAW_DIR / "clinvar"

# 3-letter -> 1-letter amino acid codes.
AA3TO1 = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C", "Gln": "Q",
    "Glu": "E", "Gly": "G", "His": "H", "Ile": "I", "Leu": "L", "Lys": "K",
    "Met": "M", "Phe": "F", "Pro": "P", "Ser": "S", "Thr": "T", "Trp": "W",
    "Tyr": "Y", "Val": "V",
}
# p.Arg175His  (single-residue missense only; excludes Ter/del/fs)
P_CHANGE = re.compile(r"p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})\b")

# Map raw ClinVar germline classifications to a clean binary; drop everything else.
PATHOGENIC = {"Pathogenic", "Likely pathogenic", "Pathogenic/Likely pathogenic"}
BENIGN = {"Benign", "Likely benign", "Benign/Likely benign"}


def _get(endpoint: str, params: dict) -> requests.Response:
    params = {**params, "tool": TOOL, "email": EMAIL}
    resp = requests.get(f"{EUTILS}/{endpoint}", params=params, timeout=90)
    resp.raise_for_status()
    time.sleep(0.34)  # stay under NCBI's 3 req/s unauthenticated limit
    return resp


def _esearch_ids(gene: str, sig: str) -> list[str]:
    term = (
        f'{gene}[gene] AND "missense variant"[molecular consequence] '
        f"AND {sig}[clinical significance]"
    )
    ids: list[str] = []
    retstart, retmax = 0, 500
    while True:
        res = _get("esearch.fcgi", {
            "db": "clinvar", "term": term, "retmode": "json",
            "retstart": retstart, "retmax": retmax,
        }).json()["esearchresult"]
        batch = res.get("idlist", [])
        ids.extend(batch)
        retstart += retmax
        if retstart >= int(res.get("count", 0)) or not batch:
            break
    return ids


def _esummaries(ids: list[str]) -> dict:
    out: dict = {}
    for i in range(0, len(ids), 200):
        chunk = ids[i : i + 200]
        res = _get("esummary.fcgi", {
            "db": "clinvar", "id": ",".join(chunk), "retmode": "json",
        }).json()["result"]
        out.update({k: v for k, v in res.items() if k != "uids"})
    return out


def fetch_clinvar_missense(gene: str, refresh: bool = False) -> list[dict]:
    """Return cleanly-labeled pathogenic/benign single-residue missense for a gene.

    Each row: ``gene, hgvs_p, position, wt_aa, mut_aa, clinsig, path_label``.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    cache = CACHE / f"{gene}.json"
    if cache.exists() and not refresh:
        return json.loads(cache.read_text())

    rows: list[dict] = []
    seen: set[tuple] = set()
    for sig in ("pathogenic", "benign"):
        ids = _esearch_ids(gene, sig)
        summaries = _esummaries(ids)
        for rec in summaries.values():
            title = rec.get("title", "")
            classification = rec.get("germline_classification", {}).get("description", "")
            if classification in PATHOGENIC:
                label = 1
            elif classification in BENIGN:
                label = 0
            else:
                continue  # drop Conflicting / Uncertain
            m = P_CHANGE.search(title)
            if not m:
                continue
            wt3, pos, mut3 = m.group(1), int(m.group(2)), m.group(3)
            if wt3 not in AA3TO1 or mut3 not in AA3TO1:
                continue  # excludes Ter (stop), etc.
            wt, mut = AA3TO1[wt3], AA3TO1[mut3]
            key = (pos, wt, mut)
            if key in seen:
                continue
            seen.add(key)
            rows.append({
                "gene": gene, "hgvs_p": f"p.{wt3}{pos}{mut3}", "position": pos,
                "wt_aa": wt, "mut_aa": mut, "clinsig": classification,
                "path_label": label,
            })
    cache.write_text(json.dumps(rows, indent=2))
    return rows
