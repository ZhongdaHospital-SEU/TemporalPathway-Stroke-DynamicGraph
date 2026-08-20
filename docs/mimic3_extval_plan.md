# MIMIC-III Full-Ordinal External Validation Plan (2026-08-17)

## Purpose
Validate the full six-level ordinal outcome framework and the harmonized feature schema on MIMIC-III, a different patient cohort (2001-2012), ICD coding system (ICD-9 430-438 vs ICD-10 I60-I64), and ICU documentation era. This directly validates the study's core ordinal claim in an external cohort, complementing the eICU external validation.

## Why this is stronger than AmsterdamUMCdb
- MIMIC-III has discharge disposition -> full 6-level ordinal outcome replicable.
- Different era and ICD coding -> answers "do results depend on coding/era".
- Pipeline reuse is high (MIMIC-IV builders adapt at the loading layer only).

## Data access (user action required)
1. PhysioNet -> project "MIMIC-III Clinical Database v1.4" -> Request access -> accept DUA (CITI training already completed: citiCompletionCertificate_15859616_78884170.pdf).
2. Download the CSV archive (mimic-iii-clinical-database-1.4.zip, approx 6.7 GB) or individual CSVs.
3. Place files in: D:\TT paper\0811Temporal Pathway\mimic-iii-1.4\ (gz files kept as-is).
Required tables: ADMISSIONS.csv.gz, PATIENTS.csv.gz, ICUSTAYS.csv.gz, D_ICD_DIAGNOSES.csv.gz, DIAGNOSES_ICD.csv.gz, D_ITEMS.csv.gz, CHARTEVENTS.csv.gz, D_LABITEMS.csv.gz, LABEVENTS.csv.gz.

## Cohort design
- Stroke admissions: any ICD-9 code 430-438 in DIAGNOSES_ICD (primary or secondary), mirroring the eICU wide definition.
- Primary analysis: admissions with admittime year 2001-2007 ONLY (era-disjoint from MIMIC-IV, which starts 2008; avoids patient overlap since MIMIC-IV 3.1 has no mimic_iii_subject_id mapping column).
- Sensitivity analysis: all MIMIC-III stroke admissions (2001-2012), with overlap noted as limitation.
- Unit of analysis: admission (hadm_id); use first ICU stay (icustay_id) per admission, mirroring the MIMIC-IV pipeline.
- Modeling cohort: hadm_id with usable ICU sequence (>=4 h coverage), complete harmonized static features, and non-missing ordinal outcome.

## Outcome mapping (6-level, mirrors MIMIC-IV)
- HOME -> 0
- HOME HEALTH CARE -> 1
- REHAB (discharge_location contains REHAB) -> 2
- SNF -> 3
- HOSPICE -> 4
- hospital_expire_flag == 1 (or discharge_location DEAD/DIED) -> 5
- Other locations (LONG TERM CARE, PSYCH FACILITY, ACUTE HOSPITAL, OTHER, AMBULATORY SURGERY CENTER, AGAINST ADVICE, INTERNAL TRANSFER, etc.) -> excluded (same as MIMIC-IV mapping).

## ICU sequences (64 x 8, same schema as harmonized MIMIC-IV/eICU)
From CHARTEVENTS, itemid map:
- hr: 211, 220045
- rr: 618, 220210
- spo2: 646, 220277
- sbp: 51, 455, 220179, 220050
- dbp: 8441, 8440, 220180, 220051
- mbp: 52, 456, 220181, 220052
- temp: 678, 679, 223761, 223762
- gcs: 198, 220739 (GCS total); fallback to component sum 223900+223901+223902 when total missing
Anchor at ICU intime, hourly resampling to 64 steps, same as build_harmonized_mimic.py.

## Static features (173-dim, harmonized schema)
- 19 labs x 6 stats (n, first, last, min, max, mean) from LABEVENTS mapped via D_LABITEMS labels: glucose, creatinine, potassium, sodium, chloride, bicarbonate, bun (label UREA NITROGEN), wbc, hematocrit, hemoglobin, platelets, lactate, inr, bilirubin_total, alt, ast, calcium, magnesium, anion_gap.
- 8 channels x 7 stats (first, last, min, max, mean, std, slope) from CHARTEVENTS.
- 3 demo: age (from dob/admittime, capped as in MIMIC-IV), gender_male, race_white (admissions.ethnicity).
Outputs: data/processed/mimic3_stroke/mimic3_harmonized_static.csv, mimic3_sequences.npz (seq_<hadm_id>, mask_<hadm_id>).

## Evaluation (reuse work/external_validate.py pattern)
1. XGB static and static+TS trained on MIMIC-IV (6-class) -> transfer to MIMIC-III; aggregate 6->4 classes (P0, P1+P2, P3, P4+P5); macro-AUC on ordinal-valid subset; mortality AUROC on full cohort; calibration slope/intercept for P(y>=2).
2. Deep models (DG-GRU+TPR / DG-GRU) after the harmonized retrain (Noether) -> transfer on the 64x8 sequences.
3. MIMIC-III internal 5-fold CV (4-class) for reference, mirroring eICU CV.
Outputs: results/external_validation/mimic3_*.csv / npz; numbers integrated into work/numbers_master.json, manuscript 03_results.md 3.8/3.9, Figure 3 if needed.

## Risks
- CHARTEVENTS.csv.gz approx 4.7 GB (decompressed ~30 GB): scan with duckdb streaming; do not load whole file into memory.
- GCS components need careful total/component merge.
- Lab value outliers and unit variants (mg/dL vs mmol/L) must be handled with the same rules as the MIMIC-IV harmonized builder.
- Era restriction (2001-2007) reduces the sample; expected stroke admissions approx 8-12k (to be confirmed after cohort build).
