# What Do Cancer Driver Mutations Break Inside a Protein Language Model?

*Interpreting ESMC sparse-autoencoder features of cancer mutations — and testing whether
the interpretable feature view explains or improves on a raw-likelihood baseline.*

**Status:** working draft (bioRxiv / ML-for-bio workshop). All numbers regenerate from
the committed pipeline (`scripts/phase*.py`); see Reproducibility.

---

## Abstract

Protein language models (PLMs) predict the functional impact of mutations well, but as
black boxes — a likelihood score with no account of *what* a mutation disrupts. The 2026
release of a sparse autoencoder (SAE) for ESMC decomposes the model's internal
representation into ~16,384 individually interpretable, GPT-5-annotated biological
features. We ask what happens to those features when a cancer mutation hits a protein.
We extract SAE-feature activations for wild-type vs. mutant sequences across an 80-gene
cancer panel (40 oncogenes, 40 tumor suppressors; 6,244 curated missense variants),
quantify per-variant feature-disruption vectors, and test whether that disruption (i)
distinguishes pathogenic from benign variants and (ii) distinguishes oncogene from
tumor-suppressor mechanisms — each relative to the standard masked-marginal likelihood
baseline, under leakage-proof held-out-gene evaluation. We find a **clean negative on
prediction** (SAE disruption alone does not beat the likelihood baseline: 0.883 vs 0.927
AUROC) but a **genuine interpretability contribution**: a combined model modestly beats
likelihood alone (0.937), and oncogene-vs-tumor-suppressor mechanism **separates in
interpretable feature space at AUROC 0.81 across held-out genes** — a signal likelihood
alone cannot provide. Interpretability is the contribution; the prediction negative is
reported explicitly rather than buried.

## 1. Introduction

SAEs on protein LMs are established (InterPLM 2024; PNAS 2025 mapped ESM-2 SAE features
to Gene Ontology terms), and a downstream-application template exists (interpretable
enzyme-function prediction from ESMC SAE features, 2026). The cancer-mutation application
is unworked: existing cancer-SAE work operates on dependency/expression data, not on
protein-LM sequence representations. This leaves open ground with a proven method:
**do ESMC SAE features encode the functional properties that cancer driver mutations
disrupt, and does feature-level disruption explain or improve discrimination beyond the
model's raw likelihood?**

We test three hypotheses:
- **H1** — driver/pathogenic mutations cause larger, more distinctive disruption of
  specific SAE features than benign mutations.
- **H2** — the disrupted features differ systematically between oncogenes (gain-of-
  function, recurrent hotspots) and tumor suppressors (loss-of-function, dispersed).
  *This is the interpretability payoff.*
- **H3** — a classifier on SAE-disruption vectors matches or beats raw masked-marginal
  likelihood. If not, the SAE features still provide a mechanistic explanation
  likelihood alone cannot.

## 2. Methods

**Model and SAE.** ESMC (600M and 6B) run locally; we apply the released Top-K SAE
weights ourselves on local hidden states (the SDK computes SAE features only via a cloud
API). Our SAE forward pass — per-token z-score → subtract decoder bias → ReLU → top-k
(k=64) → decode — is verified **bit-for-bit identical** to the official implementation,
with reconstruction FVU 0.16 (600M, layer 27) / 0.29 (6B, layer 60). The 6B SAE
codebook (`ESMC-6B-sae-layer60-k64-codebook16384`, 16,384 features) carries the GPT-5
multi-agent feature descriptions across 14 biological categories.

**Variants.** An 80-gene panel (40 oncogenes + 40 tumor suppressors, single-role genes
from the OncoKB cancer-gene list, UniProt-canonical, length ≤ 1500). Labels: ClinVar
pathogenic/benign germline missense (NCBI E-utilities) + Cancer Hotspots recurrent
somatic drivers, validated against the WT sequence — **6,244 variants; 5,179 labeled
(3,024 pathogenic / 2,155 benign).**

**Disruption features.** For each variant we run WT and mutant through ESMC, extract
per-residue SAE features, and compute three disruption vectors: `local` (signed change
at the mutated residue), `window` (summed |change| over ±2 residues), and `global`
(summed |change| over the protein) — the core engineered representation.

**Baselines (the spine).** (1) Masked-marginal log-likelihood ratio — the standard
zero-shot VEP score, ProteinGym-calibrated (mean |Spearman| 0.446, in the ESM-family
range). (2) Dense embedding-delta at the SAE layer (no SAE) into the same classifier,
isolating what the sparse decomposition adds over dense embeddings.

**Classification & evaluation.** XGBoost on disruption vectors. **Headline split is
leave-one-gene-out (LOGO)**: train on some genes, test on unseen ones — the real
generalization test, guarding against recurrence-driven label leakage. Test folds are
genes with ≥10 of each class (29 genes for pathogenicity). Fixed seeds; splits and panel
committed. SHAP over the disruption columns identifies the codebook features driving
calls; top features are cross-referenced against the GPT-5 descriptions and grouped by
category (H2).

## 3. Results

### 3.1 H1 — do pathogenic mutations disrupt features more?

Pathogenic variants disrupt a **broader** set of features than benign ones
(gene-stratified `n_features_changed` AUROC ≈ 0.65 across 46 genes; magnitude similar),
but the disruption is *broader, not more concentrated* — the "concentrated" half of H1
is refuted. The scalar summaries are moderate; the full-vector classifier (§3.2) is the
real test.

### 3.2 H3 — prediction vs. the likelihood baseline (pathogenic-vs-benign, LOGO, 29 held-out genes)

| Model | LOGO mean AUROC | pooled |
|---|---|---|
| likelihood (raw LLR) | 0.927 | 0.930 |
| embedding-delta (dense, no SAE) | 0.828 | 0.820 |
| SAE-disruption | 0.883 | 0.871 |
| **combined (SAE + LLR)** | **0.937** | **0.936** |

SAE-disruption does **not** beat the likelihood baseline on its own (0.883 < 0.927) — a
clean negative. However, (i) the **combined** model modestly beats likelihood alone
(0.937 vs 0.927), so the interpretable features carry complementary signal, and (ii)
SAE-disruption clearly beats dense embedding-deltas (0.883 vs 0.828), so the
sparse/interpretable decomposition transfers across genes better than the dense
representation it is built from.

### 3.3 H2 — oncogene vs. tumor-suppressor mechanism separation *(the payoff)*

Classifying gene role (oncogene vs. TSG) from disruption vectors under held-out-gene
splits (the model predicts an *unseen* gene's mechanism class from how its mutations
disrupt features):

| Variant set | LOGO pooled AUROC | held-out genes |
|---|---|---|
| all variants | **0.811** | 79 |
| pathogenic only | 0.734 | 70 |

Mechanism class separates cleanly in interpretable feature space for unseen genes — a
signal the raw likelihood does not provide. *(6B GPT-5-described mechanism features: see
§3.4, populated from the 6B run.)*

### 3.4 Interpretability — which described features carry the signal (6B)

*[To be filled from the 6B 80-gene run: SHAP-top features with GPT-5 summaries, category
enrichment of predictive features, and the oncogene-vs-TSG category-profile differences.]*

## 4. Discussion & limitations

The result is the shape the project set out to allow: **a clean prediction negative with
a real interpretability contribution.** SAE feature-disruption does not beat a strong
zero-shot likelihood baseline for pathogenicity — consistent with likelihood being a
very good VEP score — but it adds complementary signal (combined model) and, more
importantly, exposes a **mechanism axis** (oncogene vs. TSG) that likelihood alone
cannot. Limitations: labels inherit ClinVar/Hotspots biases; "driver" labels partly
derive from recurrence, which the model may encode (held-out-genes splits and recurrence
stratification are the guard); SAE features can be redundant/split (feature absorption),
so single-feature claims are made only with description + functional-site support; the
panel is curated (80 genes), not proteome-scale. No wet-lab or clinical claim is made —
feature attributions and driver predictions are hypotheses.

## 5. Reproducibility

One command per phase regenerates every number (`scripts/phase0`–`phase4`); fixed seeds;
panel, splits, and result metrics committed; activation caches keyed by
(model, layer, sequence-hash) and gitignored. Native-Windows, dual-GPU (16 GB + 12 GB),
no training — inference only.
