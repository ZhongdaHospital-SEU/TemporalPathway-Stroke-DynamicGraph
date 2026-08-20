# Paper Outline

**Title:** Temporal Pathway-Regularized Dynamic Graph Learning for Stroke Brain Remodeling and Ordinal Recovery Prediction

**Target journal:** Briefings in Bioinformatics (BIB) or Computational and Structural Biotechnology Journal (CSBJ), CAS Q1.
**Status:** data pipeline complete; model results final; manuscript sections pending.

## 1. Introduction
- Stroke recovery is heterogeneous; ordinal discharge disposition is a pragmatic, routinely recorded recovery proxy.
- ICU longitudinal monitoring (vitals, GCS, labs) captures acute trajectory; static admission features miss dynamic risk.
- Blood pathway activity remodels over acute (3h-24h) and subacute (baseline-follow-up) phases; pathway-level priors can regularize temporal deep models.
- Gap: no framework integrates temporal clinical sequences with cross-cohort pathway-remodeling priors for ordinal recovery prediction.
- Contribution: (i) MIMIC-IV stroke cohort with 64-step left-aligned ICU sequences and 6-level ordinal outcome; (ii) temporal pathway atlas from 4 GEO stroke cohorts via ssGSEA; (iii) DG-GRU + temporal pathway regularization (TPR); (iv) stacked ensemble with clinical XGBoost, showing significant macro-AUC gain and interpretable pathway evidence.

## 2. Methods
### 2.1 Data
- MIMIC-IV v3.1: stroke admissions (n=4858; ICU-sequence subset n=2875), 64-step left-aligned ICU time series (14 features: GCS eye/verbal/motor, vitals, labs), static features (n=218: demographics, comorbidities, labs summaries).
- Outcome: ordinal discharge disposition HOME(0) < HOME CARE(1) < REHAB(2) < SNF(3) < HOSPICE(4) < DIED(5).
- Missingness: ~69% NaN within valid windows; per-fold standardization + ffill/bfill + mean fill.
- GEO cohorts: GSE37587 (34x2 paired baseline/follow-up), GSE58294 (92 samples 3h/5h/24h cardioembolic), GSE16561, GSE22255.

### 2.2 Temporal pathway atlas (GEO)
- ssGSEA activity per sample for KEGG / Reactome / GO-BP gene sets.
- Paired t-tests: GSE37587 Baseline vs Follow-Up (subacute), GSE58294 3h vs 24h (acute); BH-FDR.
- Result: 70 significant pathways in subacute phase (FDR<0.05); acute phase no pathway survives paired FDR -> reported transparently.

### 2.3 Models
- XGBoost static-only, static+TS baselines (5-fold stratified OOF).
- GRU ordinal cumulative-link head; GRU+TPR; DG-GRU (adaptive adjacency from pathway co-membership); DG-GRU+TPR.
- TPR: pathway-prior regularizer on hidden representations (temporal pathway prior).
- Stacking (level-1 logistic, C=0.1, nested CV): XGB static + XGB static+TS + DG-GRU+TPR.

### 2.4 Evaluation
- Metrics: accuracy, Cohen's kappa, macro-AUC, Somers' D, Brier.
- Paired tests: bootstrap 95% CI for macro-AUC diff; DeLong per class.
- Clinical utility: category-free NRI/IDI (ICU timeseries increment; pathway-prior increment), DCA net benefit, calibration slope/R.

## 3. Results
- Headline: Stack AUC 0.8304 vs XGB static+TS 0.8260 (95% CI [+0.0005, +0.0084]); class 0 p=0.048, class 4 p=0.047 (DeLong).
- Deep models alone (GRU/DG-GRU +/- TPR) underperform XGB; complementary signal captured by stacking.
- Ablation: GRU 0.8139 / GRU+TPR 0.8152 / DG-GRU 0.8136 / DG-GRU+TPR 0.8147.
- Feature groups (XGB gain): Labs 0.48, GCS 0.19, Vitals 0.15, Motor 0.09.
- GEO: subacute atlas 70 FDR-significant pathways (e.g., dopaminergic neuron differentiation down, nucleic acid-templated transcription up); heatmap + trajectories.
- Clinical: ICU-timeseries increment mean NRI +0.19; Stack improves DCA for death risk (y>=5) and calibration R>=0.98.

## 4. Discussion / Limitations
- Discharge disposition as surrogate; no mRS in MIMIC-IV.
- Cohort imbalance across 6 ordinal classes; ICU-sequence subset selection bias.
- Pathway prior derived from peripheral blood GEO; transfer to ICU clinical trajectories is indirect.
- v2 attention variants overfit (AUC 0.796); discarded.
- Future: external validation (e.g., eICU), imaging features, mRS-labeled cohorts.

## 5. Figures/Tables Plan
- Figure 1: study overview + model architecture (TBD Figure 2).
- Figure 2: cohort flow + temporal pathway atlas heatmap.
- Figure 3: pathway trajectories (acute/subacute).
- Figure 4: model comparison bar (done: model_comparison.png).
- Figure 5: SHAP importance (done) + recovery trajectories by outcome (done).
- Figure 6: DCA curves + calibration.
- Table 1: cohort characteristics by outcome (done: table1.csv).
- Table 2: model comparison (acc/kappa/AUC).
- Table 3: ablation.
- Table 4: NRI/IDI/DCA/calibration.
- Supplementary: per-class DeLong, atlas full list.

## Artifacts (results/)
- stack_oof.npz (y, xgb_ts, dggru_tpr, stack1, stack2, proba)
- clinical_eval.txt, delong_result.txt, stacking_result.txt, table1.csv
- figures/: model_comparison.png, pathway_atlas_heatmap.png, pathway_trajectories.png, recovery_trajectories.png, shap_importance.png
