"""GPT-5 (multi-agent) feature descriptions for the ESMC-6B SAE codebook.

Source: HuggingFace dataset ``biohub/ESMC-SAE-Features`` — 16,384 rows, one per feature
of ``ESMC-6B-sae-layer60-k64-codebook16384`` (the descriptions used to build the ESM
Atlas). Cached to a slim CSV under ``data/raw/`` so later use (alongside torch) never
imports pyarrow after torch (Windows access-violation guard).

These map our SHAP-ranked ``feature_id``s to interpretable biology for H2 — and only
the 6B SAE has them, which is why H2 requires the 6B run.
"""

from __future__ import annotations

# pyarrow before any torch import (Windows DLL guard); this module pulls no torch.
import pyarrow.parquet  # noqa: F401

import pandas as pd
from huggingface_hub import hf_hub_download

from ..variants.panel import RAW_DIR

DESC_REPO = "biohub/ESMC-SAE-Features"
DESC_FILE = "uniref90_feature_table.parquet"
SLIM_CSV = RAW_DIR / "esmc6b_feature_descriptions.csv"
SLIM_COLS = [
    "feature_id", "summary", "category", "description",
    "activation_pattern", "exemplar_protein_families",
    "uniref90_frequency", "uniref90_idf", "uniref90_max_activation",
]


def fetch_descriptions(refresh: bool = False) -> pd.DataFrame:
    """Download the full table and cache a slim CSV keyed by feature_id."""
    if SLIM_CSV.exists() and not refresh:
        return pd.read_csv(SLIM_CSV)
    path = hf_hub_download(DESC_REPO, DESC_FILE, repo_type="dataset")
    df = pd.read_parquet(path, columns=SLIM_COLS)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(SLIM_CSV, index=False)
    return df


def load_descriptions() -> pd.DataFrame:
    """Return the slim descriptions table indexed by feature_id (cached CSV)."""
    df = fetch_descriptions()
    return df.set_index("feature_id")
