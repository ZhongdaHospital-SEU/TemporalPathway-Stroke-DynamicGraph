# Temporal Pathway-Regularized Dynamic Graph Learning for Ordinal Stroke Recovery Prediction from Intensive Care Time Series

Code and analysis pipelines for the manuscript:

**"Temporal Pathway-Regularized Dynamic Graph Learning for Ordinal Stroke Recovery Prediction from Intensive Care Time Series"**
(target journal: Computational and Structural Biotechnology Journal)

## Overview

The framework combines three components:

1. **ICU time-series modeling** — a 64-step hourly sequence of 14 physiological and neurological channels extracted from MIMIC-IV for a stroke cohort of 4,858 admissions (2,875 with complete ICU sequences), predicting a six-level ordinal discharge outcome.
2. **Temporal pathway atlas** — a hypothesis-generating biological prior derived from four public blood transcriptomic stroke cohorts (GSE37587, GSE58294, GSE16561, GSE22255) using single-sample gene-set enrichment; 70 subacute-phase pathways are used as the prior.
3. **DG-GRU+TPR** — a dynamic graph gated recurrent unit regularized by the pathway prior through monotone-progression and bounded-drift constraints on a latent recovery score, evaluated inside a level-1 logistic stacking ensemble.

Internal validation uses five-fold stratified cross-validation. External validation is performed in the multi-center eICU-CRD and in the earlier MIMIC-III era, and the admission-time risk signal is additionally evaluated in an independent cohort of 511 patients with recorded 90-day modified Rankin Scale outcomes. The deep temporal models are further validated in a 185-patient hospital cohort that combines ICU time series with recorded 90-day mRS outcomes.

## Repository layout

- `src/` — cohort definitions, sequence/static feature builders, GEO pathway activity, harmonization, model training (GRU, GRU+TPR, DG-GRU, DG-GRU+TPR), cross-validation, stacking, clinical evaluation, tables, figures, and DOI verification.
- `scripts/` — GEO download/parsing (R), pathway sensitivity and consistency analyses.
- `results/` — tables, figures (SVG), external-validation artifacts, bootstrap confidence intervals, calibration and reclassification outputs.
- `manuscript/` — manuscript markdown sources (front matter, methods, results, discussion, declarations).
- `docs/` — study plans, reference list, TRIPOD/STROBE strengthening report, submission documents.
- `work/numbers_master.json` — audited key numbers used by the consistency checks.
- `work/harmonized_schema.json` — the harmonized eight-channel feature schema used for external validation.

## Data access and privacy

This repository provides analysis code, the harmonized feature schema, and aggregate results only. No raw database files, clinical records, or patient-level data are included, and none will be added.

- MIMIC-IV v3.1, MIMIC-III v1.4, and eICU-CRD v2.0 are protected by the PhysioNet Data Use Agreement and cannot be redistributed. Users must obtain credentialed access at https://physionet.org, sign the relevant DUA, and place the files in the expected local paths before running the pipeline.
- The independent cohort of 511 stroke patients with recorded 90-day modified Rankin Scale outcomes and the hospital cohort of 185 stroke patients with ICU time series and recorded 90-day mRS outcomes are de-identified but remain protected by institutional ethics approval and are not redistributed. Requests should be directed to the corresponding author.
- GEO cohorts GSE37587, GSE58294, GSE16561, and GSE22255 are public and can be downloaded from NCBI GEO.

All result tables in `results/` report aggregate metrics and contain no patient identifiers.

## Reproducibility

1. Install dependencies: `pip install -r requirements.txt`
2. Download credentialed databases from PhysioNet and GEO files under `data/raw/`.
3. Run cohort and feature builders in `src/data/`, then model scripts in `src/temporal_pathway/`.
4. Key numbers are audited against `work/numbers_master.json`.

## License

To be chosen by the authors before public release. Note that PhysioNet data-use terms do not permit redistribution of the raw data.

## Citation

To be added after acceptance: authors, title, journal, DOI.
