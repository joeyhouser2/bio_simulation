"""Local application of a released ESMC sparse autoencoder.

The `esm` SDK computes SAE features only through the cloud Forge API
(`ESMCForgeInferenceClient`). The local `ESMC` model returns hidden states but never
applies the SAE. Since this project runs entirely on local GPUs (brief §9, Path A/B),
we download the released SAE weights and apply the Top-K autoencoder ourselves.

A released SAE repo (e.g. ``biohub/ESMC-6B-sae-layer60-k64-codebook16384``) contains:
    config.json          d_model, codebook_dim, k, available_layers
    layer_<L>.safetensors  W_enc, W_dec, b_dec, idf, max

Forward pass (matches the official ``_ESMCSAELayer.forward`` in the Biohub
``transformers`` fork exactly):
    x   = zscore(x)                          # per-token: (x - mean) / (std + 1e-5)
    z   = relu((x - b_dec) @ W_enc)          # pre-activations over the codebook
    f   = top_k(z, k)                        # keep k largest per residue, rest = 0
    x̂   = f @ W_dec + b_dec                  # reconstruction (in z-scored space)

`idf` / `max` implement the optional TF-IDF feature normalization that the Forge API
exposes as ``normalize_features`` (upweights more specific features).
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import torch
from huggingface_hub import HfApi, hf_hub_download
from safetensors.torch import load_file


@dataclass
class SAE:
    """A loaded Top-K sparse autoencoder for one ESMC layer."""

    W_enc: torch.Tensor  # [d_model, codebook_dim]
    W_dec: torch.Tensor  # [codebook_dim, d_model]
    b_dec: torch.Tensor  # [d_model]
    idf: torch.Tensor  # [codebook_dim]
    max: torch.Tensor  # [codebook_dim]
    k: int
    layer: int
    d_model: int
    codebook_dim: int
    repo_id: str

    @property
    def device(self) -> torch.device:
        return self.W_enc.device

    def to(self, device: torch.device | str) -> "SAE":
        for name in ("W_enc", "W_dec", "b_dec", "idf", "max"):
            setattr(self, name, getattr(self, name).to(device))
        return self

    def zscore(self, x: torch.Tensor) -> torch.Tensor:
        """Per-token z-score over d_model — the SAE's input normalization."""
        x = x.to(self.W_enc.dtype)
        x = x - x.mean(dim=-1, keepdim=True)
        return x / (x.std(dim=-1, keepdim=True) + 1e-5)

    @torch.no_grad()
    def encode(self, x: torch.Tensor, normalize: bool = False) -> torch.Tensor:
        """Hidden states ``[..., d_model]`` -> sparse features ``[..., codebook_dim]``.

        Only ``k`` features are non-zero per residue. If ``normalize``, apply the
        TF-IDF reweighting (feature / max * idf) the Forge API uses.
        """
        xn = self.zscore(x)
        z = torch.relu((xn - self.b_dec) @ self.W_enc)
        vals, idx = z.topk(self.k, dim=-1)
        f = torch.zeros_like(z).scatter_(-1, idx, vals)
        if normalize:
            f = f / self.max.clamp_min(1e-6) * self.idf
        return f

    @torch.no_grad()
    def decode(self, f: torch.Tensor) -> torch.Tensor:
        """Sparse (un-normalized) features -> reconstructed z-scored hidden states."""
        return f @ self.W_dec + self.b_dec

    @torch.no_grad()
    def reconstruction_fvu(self, x: torch.Tensor) -> float:
        """Fraction of variance unexplained in z-scored space: ||x̃ - x̂||² / ||x̃ - mean||².

        The baseline is the global mean of the z-scored activations (~0, since z-scoring
        centers each token), matching the official ``reconstruction_loss``. A low value
        (well under ~0.3) confirms we feed the SAE the exact layer and activation
        convention it was trained on — the key correctness check when applying released
        SAE weights to locally computed hidden states.
        """
        xn = self.zscore(x)
        x_hat = self.decode(self.encode(x))
        num = (xn - x_hat).pow(2).sum()
        den = (xn - xn.mean()).pow(2).sum().clamp_min(1e-8)
        return (num / den).item()


def load_sae(repo_id: str, layer: int | None = None, device: torch.device | str = "cpu") -> SAE:
    """Download (cached) and load a released ESMC SAE from HuggingFace."""
    cfg = json.load(open(hf_hub_download(repo_id, "config.json")))
    layers = cfg["available_layers"]
    if layer is None:
        layer = layers[0]
    if layer not in layers:
        raise ValueError(f"{repo_id} has layers {layers}, requested {layer}")

    weights_file = f"layer_{layer}.safetensors"
    if weights_file not in HfApi().list_repo_files(repo_id):
        # Fall back to the single safetensors in the repo.
        weights_file = next(
            f for f in HfApi().list_repo_files(repo_id) if f.endswith(".safetensors")
        )
    sd = load_file(hf_hub_download(repo_id, weights_file))

    sae = SAE(
        W_enc=sd["W_enc"],
        W_dec=sd["W_dec"],
        b_dec=sd["b_dec"],
        idf=sd["idf"],
        max=sd["max"],
        k=cfg["k"],
        layer=layer,
        d_model=cfg["d_model"],
        codebook_dim=cfg["codebook_dim"],
        repo_id=repo_id,
    )
    return sae.to(device)
