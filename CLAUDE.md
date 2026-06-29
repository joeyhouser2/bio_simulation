# CLAUDE.md — ESMC-SAE Cancer Mutation Interpretability

Conventions and operating rules for this repo. Read alongside `initial_writeup.pdf`
(the project brief). The brief is the source of truth for scope; this file is the
source of truth for *how we work*.

## What this project is

Extract ESMC sparse-autoencoder (SAE) feature activations for wild-type vs. mutant
cancer proteins, quantify which interpretable features each mutation disrupts, and
test whether that disruption distinguishes drivers from passengers (and pathogenic
from benign) — and whether the interpretable feature view explains or improves on a
raw masked-marginal-likelihood baseline. **Interpretability is the contribution; a
clean negative on prediction is still publishable.**

## Resolved decisions (do not re-litigate)

- **Hardware path: A, with a 600M prototype lane.** The brief assumed Path B (a
  released small SAE) didn't exist — it was wrong. Biohub released downloadable,
  **ungated** SAEs for all three model scales (300M/600M/6B), every k and codebook
  size. So: **prototype the whole pipeline on ESMC-600M** (2.4 GB, runs in seconds),
  then run the flagship **ESMC-6B** (12 GB on the 16 GB card) for the real results —
  the 6B SAE is where the GPT-5 feature descriptions live. Identical code, only the
  model + SAE repo ids change. Path C (train our own SAE) is unnecessary.
- **The local SDK does NOT apply the SAE.** `esm`'s local `ESMC.logits()` returns
  hidden states only; SAE features come from the cloud Forge API. We apply the SAE
  ourselves on local hidden states (`src/sae_cancer/esmc/sae.py`) — verified
  **bit-for-bit identical** to the official `transformers` `_ESMCSAELayer`.
- **SAE forward = z-score(x) → (x − b_dec) → relu → top-k → decode.** Input is
  per-token z-scored over d_model. **Layer convention: SAE "layer N" = hidden_states
  [N − 1]** (600M layer 27 → index 26; 6B layer 60 → index 59). Confirmed by
  reconstruction-FVU minimum (~0.16, i.e. ~84% variance explained).
- **GPUs:** RTX 4060 Ti **16 GB** (torch `cuda:1`, idle — dedicate to the model) +
  RTX 4070 SUPER 12 GB (`cuda:0`, drives display). `pick_device()` auto-routes to the
  most-free card. 6B in bf16 ≈ 12 GB fits the 4060 Ti; no quantization needed.
- **Environment: native Windows** (PowerShell), Python 3.12 venv via `uv`, torch
  2.6 cu124. No WSL, no bitsandbytes/triton.
- **SAE codebook:** `biohub/ESMC-6B-sae-layer60-k64-codebook16384` (16,384 features,
  Top-K=64, GPT-5 annotated) — the variant named in the brief; 600M analogue is
  `biohub/ESMC-600M-sae-layer27-k64-codebook16384`.
- **Windows gotcha:** the esm dep pulls a custom `transformers` fork (~5,200 files);
  installing needs `git config --global core.longpaths true` or git fails with
  "Could not reset index file". A local clone lives in `vendor/esm` (gitignored).

## Engineering conventions (brief §10)

- **Reproducibility is the product.** One command regenerates every number. Fixed
  seeds. Splits committed to the repo.
- **Score everything against the raw-likelihood baseline.** SAE features must *earn*
  their complexity. If they don't beat it on prediction, say so and pivot to
  interpretation.
- **Cache SAE activations aggressively** — they are the expensive artifact. Never
  recompute silently. Key caches by `(model, layer, sequence-hash)`.
- **Held-out *genes* is the headline split.** Beware recurrence-driven label leakage.
- **Notebooks are scratch.** Anything that matters becomes tested code in `src/`.
- **Start on a 5-gene smoke panel** (KRAS, TP53, BRAF, PTEN, EGFR). Scale only once
  the full pipeline is green.
- **Interpretation claims (H2) must cite the actual GPT-5 feature description and a
  UniProt functional-site cross-check** — no hand-waving.
- **Run the model on mutants.** There is no precomputed shortcut for novel mutant
  sequences — that's why the hardware path matters.

## Workflow

- Phased delivery (brief §11). **Stop and surface findings after each phase.**
- Activation caches and raw data dumps are gitignored; commit code + splits + results.

## Layout (brief §8)

```
src/sae_cancer/
  variants/     # curation, labeling, sequence generation
  esmc/         # model loading, activation extraction, SAE decode
  disruption/   # WT->mutant feature-disruption computation
  baselines/    # masked-marginal likelihood, embedding delta
  models/       # XGBoost / Extra Trees
  interpret/    # SHAP + feature-description cross-reference
  eval/         # splits, metrics, calibration
data/raw/       # COSMIC/ClinVar/OncoKB/UniProt dumps (gitignored)
data/variants/  # curated, labeled variant table + committed splits
data/features/  # cached SAE activations (large; gitignored)
experiments/    # one yaml per run
results/
```
