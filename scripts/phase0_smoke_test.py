"""Phase 0 smoke test (brief §11).

Confirms the fully-local SAE feature-extraction path works on the user's GPU:
load ESMC -> per-residue hidden states -> apply released SAE -> 16,384-dim features.

Validates correctness via SAE reconstruction error (low FVU at the configured layer
proves we feed the SAE the activations it was trained on). Defaults to the lightweight
600M model/SAE so Phase 0 is fast; pass --model esmc_6b for the flagship path.

Run:
    .venv\\Scripts\\python.exe scripts\\phase0_smoke_test.py
    .venv\\Scripts\\python.exe scripts\\phase0_smoke_test.py --model esmc_6b
"""

from __future__ import annotations

import argparse
import time

import torch

from sae_cancer.esmc.extract import (
    MODEL_REGISTRY,
    hidden_states,
    load_esmc,
    sae_layer_index,
)
from sae_cancer.esmc.sae import load_sae

# SAE repo per model scale (k=64, codebook=16384 — the brief's configuration).
SAE_REPOS = {
    "esmc_300m": "biohub/ESMC-300M-sae-layer23-k64-codebook16384",
    "esmc_600m": "biohub/ESMC-600M-sae-layer27-k64-codebook16384",
    "esmc_6b": "biohub/ESMC-6B-sae-layer60-k64-codebook16384",
}

# A short well-characterized test protein (ubiquitin, human; UniProt P0CG48 core).
TEST_SEQ = (
    "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="esmc_600m", choices=list(MODEL_REGISTRY))
    ap.add_argument("--seq", default=TEST_SEQ)
    args = ap.parse_args()

    expected_layer = MODEL_REGISTRY[args.model]["sae_layer"]

    print(f"[1/4] Loading {args.model} ...")
    t0 = time.time()
    model = load_esmc(args.model)
    dev = model.device
    print(f"      loaded on {dev} ({torch.cuda.get_device_name(dev) if dev.type=='cuda' else 'cpu'})"
          f" in {time.time()-t0:.1f}s")

    print(f"[2/4] Forward pass on {len(args.seq)} residues ...")
    t0 = time.time()
    hs = hidden_states(model, args.seq)  # [n_layers, L, d_model]
    n_layers, L, d_model = hs.shape
    peak = torch.cuda.max_memory_allocated(dev) / 1e9 if dev.type == "cuda" else 0
    print(f"      hidden_states {tuple(hs.shape)} in {time.time()-t0:.2f}s | peak VRAM {peak:.2f} GB")

    print(f"[3/4] Loading SAE {SAE_REPOS[args.model]} ...")
    sae = load_sae(SAE_REPOS[args.model], device=dev)
    print(f"      d_model={sae.d_model} codebook={sae.codebook_dim} k={sae.k} layer={sae.layer}")
    assert sae.d_model == d_model, f"SAE d_model {sae.d_model} != model {d_model}"

    print("[4/4] Validating layer/convention via reconstruction FVU (scan around expected index) ...")
    expected_index = sae_layer_index(expected_layer)  # SAE layer N -> hidden_states[N-1]
    lo, hi = max(0, expected_index - 2), min(n_layers - 1, expected_index + 2)
    best = None
    for idx in range(lo, hi + 1):
        fvu = sae.reconstruction_fvu(hs[idx])
        flag = "  <- config (layer %d)" % expected_layer if idx == expected_index else ""
        print(f"      hidden_states[{idx:>2}]: FVU = {fvu:.4f}{flag}")
        if best is None or fvu < best[1]:
            best = (idx, fvu)

    # Extract features at the configured layer (1-indexed -> hidden_states[N-1]).
    feats = sae.encode(hs[expected_index])  # [L, codebook]
    pooled = feats.max(dim=0).values  # per-protein max-pool over residues
    nnz_per_residue = (feats > 0).sum(dim=-1).float().mean().item()
    active_features = int((pooled > 0).sum().item())

    print("\n=== RESULT ===")
    print(f"  per-residue SAE features : {tuple(feats.shape)}  (expect [*, 16384])")
    print(f"  pooled per-protein vector: {tuple(pooled.shape)}")
    print(f"  active features (pooled) : {active_features} / {sae.codebook_dim}")
    print(f"  mean nonzeros / residue  : {nnz_per_residue:.1f}  (expect ~{sae.k})")
    print(f"  best reconstruction: hidden_states[{best[0]}] (FVU {best[1]:.4f}); "
          f"config layer {expected_layer} -> index {expected_index}")

    ok = (
        feats.shape[-1] == 16384
        and abs(nnz_per_residue - sae.k) < 1.0
        and best[1] < 0.3
        and best[0] == expected_index
    )
    print(f"\n  PHASE 0 SMOKE TEST: {'PASS' if ok else 'CHECK'}")
    if not ok:
        print("  (review FVU / layer match above before scaling to 6B)")


if __name__ == "__main__":
    main()
