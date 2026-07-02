"""Local ESMC loading and hidden-state extraction.

Runs the ESMC forward pass on the user's GPU and returns per-residue hidden states at
the layer the SAE expects. The SAE itself is applied separately (see ``sae.py``).
"""

from __future__ import annotations

from pathlib import Path

import torch
from accelerate import init_empty_weights
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from safetensors.torch import load_file

from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig
from esm.tokenization import get_esmc_model_tokenizers

# Public (ungated) repo ids + architecture + the SAE's layer, per model scale.
# We load weights ourselves rather than via ESMC.from_pretrained: the 300M/600M repos
# store a single .pth nested in data/weights/, which the SDK's loader can't discover.
MODEL_REGISTRY = {
    "esmc_300m": dict(repo="biohub/esmc-300m-2024-12", sae_layer=23,
                      d_model=960, n_heads=15, n_layers=30),
    "esmc_600m": dict(repo="biohub/esmc-600m-2024-12", sae_layer=27,
                      d_model=1152, n_heads=18, n_layers=36),
    "esmc_6b": dict(repo="biohub/esmc-6b-2024-12", sae_layer=60,
                    d_model=2560, n_heads=40, n_layers=80),
}


def pick_device() -> torch.device:
    """Largest-memory CUDA device (free as tiebreak).

    Picks by *total* memory so the 6B model lands on the 16 GB 4060 Ti rather than the
    12 GB 4070 (which drives the display) — the larger card is the dedicated compute GPU.
    """
    if not torch.cuda.is_available():
        return torch.device("cpu")
    def score(i: int) -> tuple[int, int]:
        return (torch.cuda.get_device_properties(i).total_memory, torch.cuda.mem_get_info(i)[0])
    best = max(range(torch.cuda.device_count()), key=score)
    return torch.device(f"cuda:{best}")


def load_esmc(model_name: str = "esmc_600m", device: torch.device | str | None = None) -> ESMC:
    """Load a local ESMC model (bf16 on GPU). flash-attn is auto-disabled on Windows."""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"unknown model {model_name}; choose from {list(MODEL_REGISTRY)}")
    spec = MODEL_REGISTRY[model_name]
    if device is None:
        device = pick_device()
    device = torch.device(device)

    with init_empty_weights():
        model = ESMC(
            d_model=spec["d_model"],
            n_heads=spec["n_heads"],
            n_layers=spec["n_layers"],
            tokenizer=get_esmc_model_tokenizers(),
            use_flash_attn=False,  # flash_attn isn't importable on this Windows box
        ).eval()

    files = HfApi().list_repo_files(spec["repo"])
    pth = [f for f in files if f.endswith(".pth")]
    if pth:  # 300M / 600M: single nested .pth state dict (exact key match)
        sd = torch.load(hf_hub_download(spec["repo"], pth[0]), map_location="cpu",
                        weights_only=False)
        if hasattr(sd, "state_dict"):
            sd = sd.state_dict()
    else:  # 6B: sharded safetensors -> merge shards; keys carry an "esmc." prefix
        snap = Path(snapshot_download(spec["repo"]))
        sd = {}
        for shard in sorted(snap.glob("*.safetensors")):
            sd.update(load_file(str(shard)))
        sd = {k.removeprefix("esmc."): v for k, v in sd.items()}
        # the transformers 6B names the output head "lm_head"; the SDK uses "sequence_head"
        sd = {("sequence_head." + k.removeprefix("lm_head.")) if k.startswith("lm_head.")
              else k: v for k, v in sd.items()}
    # Cast to bf16 BEFORE moving to GPU (6B float32 is ~24 GB and won't fit any card);
    # assign=True materializes the init_empty_weights() meta tensors in-place.
    if device.type != "cpu":
        sd = {k: v.to(torch.bfloat16) for k, v in sd.items()}
    model.load_state_dict(sd, assign=True)
    model = model.to(device)
    return model.eval()


@torch.no_grad()
def hidden_states(model: ESMC, sequence: str) -> torch.Tensor:
    """Return all per-residue hidden states for one sequence.

    Shape ``[n_layers, L, d_model]`` including BOS/EOS positions, on the model device.
    Index ``[layer]`` is the output of transformer block ``layer`` (0-indexed) — the
    same convention the SDK/Forge ``ith_hidden_layer`` uses.
    """
    protein_tensor = model.encode(ESMProtein(sequence=sequence))
    out = model.logits(
        protein_tensor,
        LogitsConfig(sequence=True, return_hidden_states=True, ith_hidden_layer=-1),
    )
    assert out.hidden_states is not None, "model returned no hidden states"
    # [n_layers, B=1, L, D] -> [n_layers, L, D]
    return out.hidden_states[:, 0].float()


# SAE config "layer N" is 1-indexed; the SDK's hidden_states stack is 0-indexed by
# block output. Verified against the official transformers SAE: SAE layer N reconstructs
# best from hidden_states[N-1] (e.g. 600M layer 27 -> index 26, 6B layer 60 -> index 59).
def sae_layer_index(sae_layer: int) -> int:
    return sae_layer - 1


@torch.no_grad()
def single_layer_hidden(model: ESMC, sequence: str, index: int) -> torch.Tensor:
    """Per-residue hidden states at ONE (0-indexed) layer, ``[L, d_model]``.

    Memory-lean vs :func:`hidden_states`: requests only the target layer from the SDK
    (``ith_hidden_layer``) so we never materialize all 80 layers as float32 — critical
    for the 6B model, where that spike OOMs the 16 GB card.
    """
    protein_tensor = model.encode(ESMProtein(sequence=sequence))
    out = model.logits(
        protein_tensor,
        LogitsConfig(return_hidden_states=True, ith_hidden_layer=index),
    )
    assert out.hidden_states is not None, "model returned no hidden states"
    return out.hidden_states[0, 0].float()  # [1, B=1, L, D] -> [L, D]


@torch.no_grad()
def layer_hidden_state(model: ESMC, sequence: str, sae_layer: int) -> torch.Tensor:
    """Per-residue hidden states for the SAE's (1-indexed) layer, ``[L, d_model]``."""
    return single_layer_hidden(model, sequence, sae_layer_index(sae_layer))
