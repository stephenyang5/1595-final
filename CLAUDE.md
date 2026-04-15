# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**ICU Delirium Prediction** using a T-PatchGNN-inspired architecture on MIMIC-IV data. The model encodes irregular multivariate time series (IMTS) from ICU charts/labs/drugs into 8-hour patches, applies an intra-series Transformer, then an inter-series adaptive GCN for binary delirium classification.

This project runs on the **Oscar HPC cluster** (Brown University). Use the `.venv` virtual environment in this directory. Whenever running operations, work in an interactive session, e.g. interact -q gpu -g 1 -m 20g -n 4 -t 12:00:00 or submit slurm jobs to run code.

## Running Tests

All tests run from the **project root** (`~/1595-final/`):

```bash
# Run a specific test file
python tests/test_patch_encoder.py
python tests/test_temporal_stack.py

# Or run all tests (pytest-style if pytest is available)
python -m pytest tests/
```

Tests use only synthetic data (no MIMIC access needed).

## Data Pipeline

Raw MIMIC-IV lives at `/oscar/data/shared/ursa/mimic-iv` (override with `MIMIC_ROOT` env var). Access is controlled by Oscar group permissions.

**Step 1 — Build cohort** (run once):
```bash
python -m src.build_cohort --min-los-hours 24 -o results/cohort_icu_los_ge24.csv.gz
```
Outputs `cohort.csv` with `stay_id`, `los_hours`, `label` columns.

**Step 2 — Feature extraction**: run `01_cohort_extraction.ipynb` to produce:
- `features_hourly.csv` — long-format, post-LOCF (stay_id, hour_offset, feature_name, value)
- `features_hourly_prelocf.csv` — same schema but **before** LOCF filling; used to set faithful `point_mask`

**Step 3 — EDA/Table1**: `02_eda.ipynb`, `03_table1.ipynb`

## Architecture (`src/models/`)

```
PatchTTCNEncoder          (patch_encoder.py)
  LearnableTimeEmbedding  (time_embedding.py)   — linear + periodic time encoding
  TTCN                    (ttcn.py)             — meta-filter conv over each patch
      → (B, V, P, hid_dim) patch embeddings

TemporalAdaptiveGNNStack  (temporal_adaptive_stack.py)
  PositionalEncoding      (positional_encoding.py)
  TransformerEncoder      — intra-series: long-range deps within each variable
  Adaptive GSL + GCN      (gcn.py)              — inter-series: dynamic variable graph
      → (B, V, P, hid_dim) contextual embeddings

DeliriumTPatchBackbone    (delirium_backbone.py)
  = PatchTTCNEncoder + TemporalAdaptiveGNNStack
  → (B, V, P, hid_dim) [classification head not yet attached]
```

**Key dimensions**: `B`=batch, `V`=54 variables, `P`=num patches, `L`=8 (patch_hours), `D`=hid_dim (default 32).

## Data Layer (`src/data/`)

- `feature_vocab.py` — canonical list of **54 features** (14 chart + 23 lab + 17 drug). Order is fixed; never reorder.
- `patch_dataset.py` — `ICUPatchDataset` builds `(V, P, L)` tensors per stay. If `features_hourly_prelocf.csv` is present alongside the features file, it is auto-loaded to set faithful `point_mask` (marking only true observations, not LOCF-filled hours).
- `collate_patches()` — pads `P` and `L` dims to batch maxima; use as `DataLoader(collate_fn=collate_patches)`.

## Reference Papers

### T-PatchGNN (zhang24bw.pdf) — architecture source

The `src/` model stack is a direct adaptation of **T-PatchGNN** (Zhang et al., ICML 2024) for binary classification instead of forecasting. Code-to-paper mapping:

| File | Paper component |
|------|----------------|
| `src/models/ttcn.py` | TTCN meta-filter (Eq. 5–6): adaptively-generated conv filter scaling to variable-length patches |
| `src/models/time_embedding.py` | Continuous time embedding φ(t) (Eq. 3): learnable linear + sinusoidal terms |
| `src/models/patch_encoder.py` | Patch Encoding block (Fig. 2b): TTCN output + binary patch-mask term concatenated |
| `src/models/temporal_adaptive_stack.py` | Intra/Inter-time series modeling block (Fig. 2c): Transformer over patches + time-varying adaptive GSL + GCN |
| `src/models/gcn.py` | GNN for inter-series correlation (Eq. 11) |
| `src/models/delirium_backbone.py` | Full IMTS_Model backbone (forecasting MLP head omitted; classification head is the next milestone) |

Key design rationale from the paper:
- **Patch size 8h**: the paper validated 8h as optimal for MIMIC-scale sparse physiological signals.
- **Adaptive graph** (Eq. 9–10): per-patch adjacency `A_p` is built by gating static node embeddings (`nodevec1/2`) with time-varying patch features — this is what `nodevec_gate1/2` and `nodevec_linear1/2` implement in `TemporalAdaptiveGNNStack`.
- **`stay_patch_mask`**: prevents batch-padding patches (beyond a patient's actual LOS) from contributing to Transformer attention.

### DeLLiriuM (nihpp-rs7216692v1.pdf) — clinical standard and evaluation target

**DeLLiriuM** (Contreras et al., 2025) is the current SOTA for ICU delirium prediction and sets the evaluation standard.

**Delirium definition** — must match for a fair comparison:
- CAM-ICU positive **and** RASS ≥ −3 at any 12-hour assessment interval occurring **after** the first 24 hours of ICU stay.

**Cohort inclusion/exclusion criteria** (`src/cohort.py` / `src/build_cohort.py` must enforce these):
- First ICU admission only
- Age ≥ 18
- ICU LOS ≥ 24 hours
- Exclude: coma or delirium present in the **first 24h** (goal is predicting onset, not screening prevalent cases)
- Exclude: death within 48h of admission
- Exclude: missing data in first 24h

**MIMIC-IV incidence**: ~6.8% delirium — class imbalance justifies weighted BCE loss.

**Target metrics** (use same protocol as DeLLiriuM for comparability):
- Primary: **AUROC** — report median + 95% CI via 200-iteration bootstrap
- Secondary: **AUPRC** — critical given ~6.8% positive rate
- Baselines to beat: best structured EHR deep learning model (Transformer) achieved AUROC ~78.1 on external validation; DeLLiriuM (345M-param LLM) achieved ~82.5.

**Top predictive features** per SHAP (informing feature vocabulary priorities):
- Assessment scores: GCS, RASS, CAM (highest importance across all cohorts)
- Respiratory: PEEP, tidal volume, SpO2
- Labs: BNP, anion gap, lactic acid, creatinine, specific gravity (urine)
- Vital signs: heart rate, blood pressure

## Important Implementation Notes

- **`point_mask` vs `patch_mask` vs `stay_patch_mask`**: `point_mask (B,V,P,L)` marks real observations within a patch; `patch_mask (B,V,P)` is 1 if any observation in that variable-patch; `stay_patch_mask (B,P)` is 1 if the patch is within the ICU stay (not batch padding).
- **Adaptive graph**: per-patch adjacency `A_p` is constructed dynamically by gating static node embeddings with time-varying features. Static prior graphs can be passed via `static_supports` to `TemporalAdaptiveGNNStack`.
- **Classification head**: `DeliriumClassifier` wraps `DeliriumTPatchBackbone` with masked mean pooling over patches → mean over variables → `Dropout + Linear(D, 1)`. Use `BCEWithLogitsLoss`; sigmoid is applied externally at evaluation.
- **MIMIC paths**: all MIMIC table paths are centralized in `src/mimic_paths.py`. If running outside Oscar, set `MIMIC_ROOT`.
