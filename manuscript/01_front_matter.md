# Temporal Pathway-Regularized Dynamic Graph Learning for Ordinal Stroke Recovery Prediction from Intensive Care Time Series

*Manuscript draft v0.2 (2026-08-17)*, target journal: *Computational and Structural Biotechnology Journal* (CSBJ)

---

## Title page

**Article type:** Research article

**Title:** Interpretable Temporal Pathway-Regularized Dynamic Graph Learning for Ordinal Stroke Recovery Prediction and Cross-System Transportability

**Running title:** Interpretable pathway-regularized stroke recovery prediction

**Authors:** [Author 1], [Author 2], [Author 3], [Author 4], [Corresponding Author]

**Affiliations:** [Department/Institution 1]; [Department/Institution 2]; [to be completed]

**Corresponding author:** [Name], [Email], [Address], [to be completed]

**Word count (main text):** 6665
**Abstract word count:** 345
**Figures:** 5 (Figure 1A–D, Figure 2A–G, Figure 3A–D, Figure 4A–D, Figure 5A–D); **Tables:** 8

**Keywords:** stroke; ordinal outcome; intensive care unit; deep learning; graph neural network

**Conflicts of interest:** The authors declare no competing interests. [to be confirmed]

**Funding:** [to be completed]

---

## Abstract

Stroke recovery is heterogeneous, and early identification of patients at risk of poor functional outcome is important for discharge planning and rehabilitation allocation. Routinely recorded hospital discharge disposition provides a pragmatic ordinal proxy of recovery in electronic health record databases, yet existing models either ignore the acute-phase physiological trajectory or omit biological prior knowledge. We developed an interpretable temporal pathway-regularized dynamic graph learning framework that combines a 64-step intensive care unit (ICU) time series of 14 physiological and neurological channels, a hypothesis-generating temporal pathway atlas derived from four public blood transcriptomic stroke cohorts, and a pathway-prior-regularized dynamic graph gated recurrent unit (DG-GRU+TPR) that models temporal dynamics under monotone-progression and bounded-drift constraints. Using 4,858 MIMIC-IV stroke admissions, including 2,875 with complete ICU sequences, and a six-level ordinal discharge outcome, the stacked ensemble of gradient-boosted static and temporal models with DG-GRU+TPR achieved a macro-AUC of 0.830, exceeding the strongest gradient-boosting baseline at 0.826 (bootstrap 95% CI of the difference +0.0005 to +0.0083). Quantified sensitivity analyses showed that the pathway prior contributes interpretable structure rather than a clinically meaningful predictive gain: the regularized latent score responded monotonically to in-silico physiological deterioration, its structural properties were reproducible across random seeds, and it carried prognostic information earlier in the stay than the unregularized score, while negative-control and leakage audits confirmed that the results are free of label or temporal leakage. External validation preserved discrimination in the multi-center eICU-CRD with a four-class macro-AUC of 0.809 and mortality AUROC of 0.962, and in MIMIC-III with a four-class macro-AUC of 0.814 and mortality AUROC of 0.920, with stable results under a narrow ischemic-stroke definition and across admission eras; deep-model transfer to eICU matched the harmonized development estimates, with four-class macro-AUC values of 0.807 to 0.816. In an independent cohort of 511 patients with recorded 90-day modified Rankin Scale outcomes, the admission-time risk model transferred with an AUROC of 0.635, with an internal cross-validation AUROC of 0.739, confirming partial transportability of the risk signal to a functional outcome. The framework provides a transferable template for integrating cross-cohort biological priors with EHR-based temporal deep learning, in which the pathway prior is positioned explicitly as an interpretability mechanism rather than a predictive increment.

---

## 1. Introduction

Stroke remains a leading cause of adult disability and death worldwide, and the trajectory of recovery after the acute event is highly variable [1,2]. Some patients return to independent living within weeks, whereas others require long-term institutional care or die during the index hospitalization [3]. Early and accurate risk stratification is therefore central to discharge planning, rehabilitation intensity, and advance-care discussions [4,5]. A quantitative model that estimates the expected recovery level at the time of ICU admission could help clinicians deploy resources to the patients who need them most.

Two complementary sources of information can inform such a model. First, the electronic health record, abbreviated EHR, provides a dense longitudinal stream of vital signs, neurological assessments, and laboratory values during the ICU stay, which reflects the acute physiological trajectory that static admission variables cannot capture [6,7]. Machine-learning models applied to such temporal features have consistently outperformed static-only models for in-hospital mortality and clinical deterioration [8,9]. Second, stroke elicits a measurable systemic biological response. Peripheral blood transcriptomic studies have repeatedly documented coordinated remodeling of immune, inflammatory, and neural-repair pathways across the acute phase, which spans hours, and the subacute phase, which spans days to weeks [10,11]. Public gene-expression cohorts such as GSE37587 and GSE58294 therefore encode a temporal pathway atlas of stroke recovery that is independent of any single hospital's records [12,13].

Current approaches leave three gaps. Most EHR-based stroke outcome models treat the admission as a static snapshot and discard the temporal dynamics that are most informative for recovery [14]. Deep sequential models that do consume ICU time series are typically difficult to interpret, and their training does not incorporate domain knowledge about the biological processes relevant to recovery [15,16]. Finally, although pathway-level priors have been used to regularize deep models in genomics [17,18], they have not been combined with EHR-derived clinical time series for ordinal functional-outcome prediction.

In this study, we propose an interpretable temporal pathway-regularized dynamic graph learning framework that addresses these gaps. We first constructed a MIMIC-IV stroke cohort of 4,858 admissions with a six-level ordinal discharge outcome and 2,875 patients with complete 64-step ICU time series and 218 static features. We then derived a temporal pathway atlas from four public stroke transcriptomic cohorts using single-sample gene-set enrichment, identifying 70 subacute-phase pathways with significant paired activity change in the derivation cohort. We treat this atlas as hypothesis-generating: it provides a mechanism-motivated prior for the clinical model, while its cross-cohort generalization is explicitly tested rather than assumed. We developed a dynamic graph gated recurrent unit regularized by this pathway prior, termed DG-GRU+TPR, and evaluated it within a stacked ensemble. Finally, we subjected the entire pipeline to a battery of robustness, negative-control, and leakage-audit analyses, so that the reported findings are accompanied by explicit evidence of their boundaries. The primary aim of this study was to deliver an ordinal stroke recovery model whose discrimination transfers across health systems and whose internal representation remains clinically interpretable; the secondary aim was to characterize precisely what the pathway prior contributes, rather than to claim an unqualified predictive increment. We hypothesized that encoding the prior as monotone-progression and bounded-drift constraints would reshape the latent recovery trajectory toward an interpretable, outcome-ordered progression while preserving discrimination, and that this structure, rather than a predictive gain, would constitute the prior's value.