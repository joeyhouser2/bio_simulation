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
- **Next:** Phase 3 — classify on disruption vectors vs both baselines; SHAP interpret.

```powershell
.venv\Scripts\python.exe scripts\phase1_curate.py        # build the variant table
.venv\Scripts\python.exe scripts\phase1_baseline.py      # likelihood baseline + metrics
.venv\Scripts\python.exe scripts\phase1_calibration.py   # ProteinGym calibration
.venv\Scripts\python.exe scripts\phase2_disruption.py    # disruption features + H1
```
