"""ProteinGym calibration of the masked-marginal likelihood baseline (brief §5.3).

Before comparing SAE features to the likelihood baseline, confirm the baseline is
"known-good": its masked-marginal LLR should correlate with experimental DMS fitness on
ProteinGym. We use the assays overlapping the panel genes (TP53/PTEN/KRAS) and report
the per-assay Spearman |rho| — the standard ProteinGym metric.

Data: OATML-Markslab/ProteinGym_v1 (HuggingFace). Each row carries the assay's
reference ``target_seq``, so mutants are scored against the assay's own sequence.
"""

from __future__ import annotations

# IMPORTANT: pyarrow must be imported before torch on Windows, or reading parquet after
# torch is loaded triggers a native access violation. Importing it at module top (this
# module pulls no torch) guarantees the safe order when imported before any torch user.
import pyarrow.parquet  # noqa: F401

import re

import pandas as pd
from huggingface_hub import hf_hub_download
from scipy.stats import spearmanr

from ..variants.panel import RAW_DIR

PG_REPO = "OATML-Markslab/ProteinGym_v1"
PG_SHARDS = [f"DMS_substitutions/train-0000{i}-of-00005.parquet" for i in range(5)]
PG_CACHE = RAW_DIR / "proteingym"

# ProteinGym assays overlapping the panel genes (UniProt entry names).
PANEL_DMS_ASSAYS = [
    "P53_HUMAN_Giacomelli_2018_WT_Nutlin",
    "P53_HUMAN_Kotler_2018",
    "PTEN_HUMAN_Matreyek_2021",
    "PTEN_HUMAN_Mighell_2018",
    "RASK_HUMAN_Weng_2022_abundance",
]

_MUT = re.compile(r"^([A-Z])(\d+)([A-Z])$")


def load_proteingym(assay_ids: list[str], refresh: bool = False) -> dict[str, pd.DataFrame]:
    """Load single-substitution rows for the requested assays, keyed by DMS_id.

    Filtered assays are cached as CSV under ``data/raw/proteingym/`` so later runs read
    plain CSV and never import pyarrow (which conflicts with torch on Windows).
    """
    PG_CACHE.mkdir(parents=True, exist_ok=True)
    out: dict[str, pd.DataFrame] = {}
    missing = []
    for aid in assay_ids:
        cache = PG_CACHE / f"{aid}.csv"
        if cache.exists() and not refresh:
            out[aid] = pd.read_csv(cache)
        else:
            missing.append(aid)

    if missing:
        want = set(missing)
        frames = []
        for shard in PG_SHARDS:
            path = hf_hub_download(PG_REPO, shard, repo_type="dataset")
            df = pd.read_parquet(path, columns=["DMS_id", "mutant", "target_seq", "DMS_score"])
            frames.append(df[df["DMS_id"].isin(want)])
        allrows = pd.concat(frames, ignore_index=True)
        allrows = allrows[~allrows["mutant"].str.contains(":")]  # single subs only
        for aid, g in allrows.groupby("DMS_id"):
            g = g.reset_index(drop=True)
            g.to_csv(PG_CACHE / f"{aid}.csv", index=False)
            out[aid] = g
    return out


def calibrate_assay(model, assay: pd.DataFrame) -> dict:
    """Spearman between masked-marginal LLR and DMS_score for one assay."""
    from ..baselines.likelihood import masked_marginal_llr  # lazy: pulls torch


    target = assay["target_seq"].iloc[0]
    variants, dms, kept = [], [], []
    for mut, score in zip(assay["mutant"], assay["DMS_score"]):
        m = _MUT.match(mut)
        if not m:
            continue
        wt, pos, alt = m.group(1), int(m.group(2)), m.group(3)
        if pos < 1 or pos > len(target) or target[pos - 1] != wt:
            continue  # mutant numbering must match the assay reference
        variants.append((pos, wt, alt))
        dms.append(float(score))
        kept.append((pos, wt, alt))
    if len(variants) < 10:
        return {"n": len(variants), "spearman": float("nan")}
    llr = masked_marginal_llr(model, target, variants)
    pred = [llr[v] for v in kept]
    rho = spearmanr(pred, dms).statistic
    return {"n": len(variants), "seq_len": len(target), "spearman": float(rho)}
