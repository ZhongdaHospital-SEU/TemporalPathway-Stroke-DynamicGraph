# Code Repository README (draft for public release)

# Temporal Pathway-Regularized Dynamic Graph Learning for Ordinal Stroke Recovery Prediction from Intensive Care Time Series

This repository contains the code, analysis pipelines, and experimental results for the manuscript:
**"Temporal Pathway-Regularized Dynamic Graph Learning for Ordinal Stroke Recovery Prediction from Intensive Care Time Series"** (target journal: Computational and Structural Biotechnology Journal).

## Overview
The framework combines (1) a 64-step hourly ICU time series of physiological and neurological channels from MIMIC-IV, (2) a temporal pathway atlas derived from four public blood transcriptomic stroke cohorts (GSE37587, GSE58294, GSE16561, GSE22255), and (3) a pathway-regularized dynamic graph gated recurrent unit (DG-GRU+TPR) that predicts a six-level ordinal discharge outcome. Models are validated internally and externally in eICU-CRD.

## Data access
- MIMIC-IV v3.1 and eICU-CRD v2.0: PhysioNet credentialed access (https://physionet.org). Users must sign the Data Use Agreement and place the files in the expected local paths before running the pipeline.
- GEO cohorts GSE37587, GSE58294, GSE16561, GSE22255: public (NCBI GEO).
- All harmonized feature schemas are defined in `work/harmonized_schema.json`.

## Repository layout
- `src/data/` — cohort definitions, sequence/static feature builders, GEO pathway activity, harmonization.
- `src/temporal_pathway/` — model training (GRU, GRU+TPR, DG-GRU, DG-GRU+TPR), cross-validation, stacking, clinical evaluation.
- `src/analysis/` — tables, figures (SVG + PNG), SHAP importance, DOI verification.
- `work/` — audit scripts, validation pipelines, consistency checks.
- `results/` — tables, figures, external-validation artifacts, bootstrap CIs.
- `manuscript/` — manuscript markdown sources and the assembled Word document.

## Reproducibility
1. Install dependencies: `pip install -r requirements.txt`
2. Download credentialed data (PhysioNet) and place GEO files under `data/raw/`.
3. Run cohort + feature builders in `src/data/`, then model scripts in `src/temporal_pathway/`; all key numbers are audited against `work/numbers_master.json`.

## License
[To be decided before public release; the user is expected to choose an open-source license such as MIT or Apache-2.0. Note: PhysioNet data use terms do not permit redistribution of the raw data.]

## Citation
[To be added after acceptance: authors, title, journal, DOI.]
