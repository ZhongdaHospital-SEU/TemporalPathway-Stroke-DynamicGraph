# Graphical Abstract (draft)

File: results/figures/Graphical_abstract.svg (editable text) and Graphical_abstract.png (300 dpi)

Layout (left to right, four blocks):
1. **ICU time series** — 64 h hourly grid with 14 physiological and neurological channels, plus laboratory summaries (MIMIC-IV, n = 4,858 stroke admissions; modeling cohort n = 2,875).
2. **Pathway atlas** — single-sample gene-set enrichment across four public blood transcriptomic stroke cohorts (GSE37587, GSE58294, GSE16561, GSE22255); 70 subacute-phase pathways form the temporal pathway prior.
3. **DG-GRU + TPR** — dynamic feature graph feeding a gated recurrent unit; latent recovery score regularized by the pathway prior under monotone-progression and bounded-drift constraints.
4. **Ordinal outcome and validation** — six-level discharge disposition prediction; stacked ensemble macro-AUC 0.830; external validation in eICU-CRD (n = 12,820; transfer macro-AUC 0.809, mortality AUROC 0.962), stable under a narrow ischemic-stroke definition.

Status: draft sketch; final styling can be aligned with journal templates at submission time.
