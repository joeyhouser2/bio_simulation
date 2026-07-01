# ESMC-SAE Cancer Mutation Interpretability

What do cancer driver mutations break inside a protein language model? This project
extracts ESMC-6B sparse-autoencoder (SAE) feature activations for wild-type vs. mutant
cancer proteins and asks which interpretable, GPT-5-annotated features each mutation
disrupts — then tests whether that disruption separates drivers from passengers and
pathogenic from benign, relative to a raw-likelihood baseline.

See `initial_writeup.pdf` for the full brief and `CLAUDE.md` for conventions.

## Hardware path (resolved — brief §9)

**Path A, prototyped on 600M.** We run ESMC locally and apply the released SAE
ourselves (the local SDK returns only hidden states). All Biohub weights are public
and ungated. Strategy: build/iterate on **ESMC-600M** (2.4 GB, seconds per run), then
run **ESMC-6B** (bf16 ≈ 12 GB, fits the 16 GB RTX 4060 Ti — no quantization) for the
results, where the GPT-5 feature descriptions live. See `CLAUDE.md` for the full
resolved decisions and the SAE forward-pass convention.

## Setup (native Windows + CUDA)

```powershell
uv venv --python "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe"

# PyTorch with CUDA (matches the installed driver)
uv pip install torch --index-url https://download.pytorch.org/whl/cu124

# The esm package pulls a large custom transformers fork — Windows needs long paths:
git config --global core.longpaths true
git clone --depth 1 https://github.com/Biohub/esm.git vendor/esm
uv pip install -e vendor/esm          # or: uv pip install "esm @ git+https://github.com/Biohub/esm.git"

# Core deps + this package
uv pip install numpy pandas scipy scikit-learn tqdm pyyaml requests biopython xgboost
uv pip install "llvmlite>=0.43" "numba>=0.60" shap
uv pip install -e . --no-deps

# Verify CUDA sees both GPUs
.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
```

## Phase 0 smoke test

```powershell
.venv\Scripts\python.exe scripts\phase0_smoke_test.py              # 600M (fast)
.venv\Scripts\python.exe scripts\phase0_smoke_test.py --model esmc_6b
```

Confirms the local path: ESMC → layer-(N−1) hidden states → SAE → `[L, 16384]`
features (k=64/residue), with reconstruction FVU ≈ 0.16 at the configured layer.

## Status

- **Phase 0 complete** — hardware path resolved, repo scaffolded, CUDA verified on
  both GPUs, local SAE extraction verified bit-for-bit against the official impl.
- **Phase 1 complete** — curated 1,275 labeled variants for the 5-gene panel
  (ClinVar pathogenic/benign + Cancer Hotspots drivers) into
  `data/variants/variants.csv`; masked-marginal likelihood baseline gives
  pathogenic-vs-benign **AUROC 0.977** overall (TP53, balanced labels: 0.960);
  ProteinGym calibration **mean |Spearman| 0.446** over 5 panel-gene assays
  (in the ESM-family zero-shot range — baseline confirmed sane).
- **Phase 2 complete** — per-variant WT→mutant SAE feature-disruption vectors
  (local / window / global), cached under `data/features/`. H1 sanity check
  (gene-stratified, confound-controlled): pathogenic variants disrupt *more* SAE
  features (`n_features_changed` AUROC 0.62, magnitude ~0.57) but the disruption is
  *broader, not more concentrated* — H1's "concentrated" half is refuted. Scalar
  signal is modest vs the likelihood baseline; the Phase-3 classifier on the full
  disruption vectors is the real test. (Pooled-across-gene scalars were confounded by
  protein length/gene — a live demonstration of why held-out-genes splits matter.)
- **Phase 3 complete (600M)** — pathogenic-vs-benign classifiers under leave-one-gene-out
  (held-out genes BRAF/EGFR/TP53). LOGO mean AUROC: **likelihood 0.975**,
  embedding-delta 0.592, **SAE-disruption 0.867**, combined 0.952. Headline findings:
  (1) the sparse SAE decomposition **generalizes across genes far better than dense
  embeddings** (0.867 vs 0.592) — what the interpretable view adds over raw embeddings;
  (2) SAE features **do not beat the raw-likelihood baseline** for held-out-gene
  prediction (the brief's anticipated clean negative — interpretability is the
  contribution). SHAP ranks the codebook features driving calls (feature 13421
  dominant) for the H2 GPT-5-description cross-reference.
- **6B run + H2 (preliminary)** — ESMC-6B loads on the 16 GB 4060 Ti (bf16, ~13 GB);
  6B disruption H1 is stronger than 600M (gene-stratified pathogenic-vs-benign:
  magnitude AUROC 0.71, n_features_changed 0.71). H2 cross-reference of SHAP-ranked
  features to GPT-5 descriptions: the most predictive disruption features skew toward
  *generic* structural/compositional/interaction features and are **depleted** for
  crisp catalytic/ligand-binding categories (against the naive "drivers hit active
  sites" story). Qualitatively, oncogene variants disrupt more Structural-motif
  features while TSG variants disrupt more Domain/Disorder features — a directional
  mechanism signal. The gene-held-out oncogene-vs-TSG classifier is unreliable at 5
  genes (degenerate folds); a powered claim needs Phase 4 scale-up.
- **Phase 4 complete (600M, 80-gene panel)** — 6,244 variants across 40 oncogenes +
  40 TSGs; 5,179 labeled pathogenic/benign. Powered results:
  - *Prediction* (pathogenic-vs-benign, leave-one-gene-out over 29 test genes):
    likelihood **0.927**, embedding-delta 0.828, SAE-disruption 0.883,
    **combined 0.937**. The combined model now *beats* likelihood alone (at 5 genes it
    hurt) — SAE features add a small but real complementary lift; SAE still doesn't beat
    likelihood on its own (clean negative holds); SAE > dense embeddings (transfer edge).
  - *H2 mechanism* (oncogene-vs-TSG from disruption vectors, held-out-gene pooled):
    **AUROC 0.811** (all variants), 0.734 (pathogenic). The 5-gene degenerate 0.21
    becomes a clean, leakage-proof 0.81 — mechanism class separates in interpretable
    feature space for *unseen* genes. **The powered interpretability payoff.**
- **Net thesis:** prediction is a near-wash vs the likelihood baseline (publishable
  clean negative), but the interpretable feature view delivers a genuine, powered
  contribution — oncogene/TSG mechanism separation and a combined-model lift.
- **Next (optional):** 6B on the 80-gene panel for GPT-5-described mechanism features;
  write-up.

```powershell
.venv\Scripts\python.exe scripts\phase1_curate.py        # build the variant table
.venv\Scripts\python.exe scripts\phase1_baseline.py      # likelihood baseline + metrics
.venv\Scripts\python.exe scripts\phase1_calibration.py   # ProteinGym calibration
.venv\Scripts\python.exe scripts\phase2_disruption.py    # disruption features + H1
.venv\Scripts\python.exe scripts\phase3_classify.py      # classifiers vs baselines (LOGO)
.venv\Scripts\python.exe scripts\phase3_shap.py          # SHAP feature attribution
```
