"""Masked-marginal log-likelihood ratio — the standard zero-shot VEP baseline.

For a missense variant (wt -> mut at position p), mask position p, run ESMC, and score:

    LLR(p, wt->mut) = log P(mut | x_\\p) - log P(wt | x_\\p)

(Meier et al. 2021). More negative => the model finds the substitution less likely =>
more deleterious. Everything in the project is reported relative to this (brief §5.2):
SAE features must earn their complexity against it.
"""

from __future__ import annotations

import torch

from esm.models.esmc import ESMC
from esm.tokenization import get_esmc_model_tokenizers

_TOK = get_esmc_model_tokenizers()
_AA_IDS = {aa: _TOK.convert_tokens_to_ids(aa) for aa in "ACDEFGHIKLMNPQRSTVWY"}
_MASK_ID = _TOK.mask_token_id


@torch.no_grad()
def masked_marginal_llr(
    model: ESMC,
    sequence: str,
    variants: list[tuple[int, str, str]],
    batch_size: int = 16,
) -> dict[tuple[int, str, str], float]:
    """Masked-marginal LLR for each ``(position, wt_aa, mut_aa)`` (1-indexed position).

    Masks each unique position once (batched), then reads off log-probabilities.
    Returns a dict keyed by the input variant tuples.
    """
    device = model.device
    base = model._tokenize([sequence])[0]  # [L+2]: <cls> ... <eos>; pos p -> index p
    positions = sorted({p for p, _, _ in variants})

    logprobs: dict[int, torch.Tensor] = {}
    for i in range(0, len(positions), batch_size):
        chunk = positions[i : i + batch_size]
        toks = base.unsqueeze(0).repeat(len(chunk), 1).clone()
        for row, p in enumerate(chunk):
            toks[row, p] = _MASK_ID
        out = model.forward(sequence_tokens=toks.to(device))
        lp = torch.log_softmax(out.sequence_logits.float(), dim=-1)  # [chunk, L+2, V]
        for row, p in enumerate(chunk):
            logprobs[p] = lp[row, p].cpu()

    scores: dict[tuple[int, str, str], float] = {}
    for p, wt, mut in variants:
        lp = logprobs[p]
        scores[(p, wt, mut)] = float(lp[_AA_IDS[mut]] - lp[_AA_IDS[wt]])
    return scores
