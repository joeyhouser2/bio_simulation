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

**Phase 0 complete** — hardware path resolved, repo scaffolded, CUDA verified on both
GPUs, local SAE extraction smoke-tested and verified bit-for-bit against the official
implementation. Next: Phase 1 (5-gene variant panel + masked-marginal likelihood
baseline).
