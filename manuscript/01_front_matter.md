# Temporal Pathway-Regularized Dynamic Graph Learning for Ordinal Stroke Recovery Prediction from Intensive Care Time Series

*Manuscript draft v0.3 (2026-08-22, synced to 0820Manuscript.docx after 1st/2nd/3rd reviewer audits)*, target journal: *Computational and Structural Biotechnology Journal* (CSBJ)

---

## Title page

**Article type:** Research article

**Title:** Temporal Pathway-Regularized Dynamic Graph Learning for Ordinal Stroke Recovery Prediction from Intensive Care Time Series

**Running title:** Temporal pathway-regularized stroke recovery prediction

**Authors:** Zhipeng Wang1*, Luning Wang2*, Changsong Wang1, Pengli Zhai3, Hui Feng1, Hongmei Liu1, Qian Hou1, Ming Guo1

**Affiliations:** 1 Department of TCM, Zhongda Hospital, Southeast University, China; 2 Department of Rehabilitation Medicine, Zhongda Hospital, Southeast University, China; 3 Jiangbei Campus, Jiangsu Provincial Traditional Chinese Medicine Hospital, China

* These authors contributed equally to this work and are co-first authors.

**Corresponding author:** Changsong Wang, Department of TCM, Zhongda Hospital, Southeast University, China. Email: 101005664@seu.edu.cn

**Figures:** 6 (Figure 1A–D, Figure 2A–G, Figure 3A–D, Figure 4A–D, Figure 5A–D, Figure 6A–D); **Tables:** 8

**Keywords:** stroke; ordinal outcome; intensive care unit; deep learning; graph neural network

**Conflicts of interest:** The authors declare no competing interests.

**Funding:** This work was supported by the Higher Education Teaching Reform Research Project of Xuzhou Medical University (XYJG042).

---

## Abstract

**Background:** Stroke recovery is heterogeneous, and early identification of patients at risk of poor functional outcome remains challenging. Existing electronic health record models either ignore the acute-phase physiological trajectory or omit biological prior knowledge.

**Methods:** We used 4,858 stroke admissions from MIMIC-IV, including 2,875 with complete 64-step ICU time series of 14 physiological and neurological channels and 218 static features, and mapped discharge disposition to a six-level ordinal outcome. A temporal pathway atlas from four public transcriptomic stroke cohorts was encoded as monotone-progression and bounded-drift constraints in a dynamic graph gated recurrent unit, DG-GRU+TPR. Models were evaluated under five-fold cross-validation in a stacking ensemble and transferred to eICU-CRD, MIMIC-III, and two independent cohorts with 90-day modified Rankin Scale outcomes.

**Results:** The stacked ensemble achieved a macro-AUC of 0.830, exceeding the strongest baseline of 0.826 with a bootstrap 95% CI of the difference from +0.0005 to +0.0083. The pathway prior contributed interpretable structure rather than a predictive gain, and negative-control and leakage audits indicated that the results were free of label or temporal leakage. Discrimination was preserved on transfer to eICU-CRD, macro-AUC 0.809 and mortality AUROC 0.962, and to MIMIC-III, 0.814 and 0.920. Transfer to a 511-patient cohort with recorded 90-day modified Rankin Scale outcomes reached an AUROC of 0.635 against an internal estimate of 0.739, and a 185-patient hospital cohort achieved 0.739 on admission-time transfer with 0.738 after fine-tuning.

**Conclusion:** The framework provides a transferable template for integrating biological priors with ICU temporal deep learning, positioning the pathway prior as an interpretability mechanism rather than a predictive increment.

---

## 1. Introduction

Stroke remains a leading cause of adult disability and death worldwide, and the trajectory of recovery after the acute event is highly variable [1,2]. Some patients return to independent living within weeks, whereas others require long-term institutional care or die during the index hospitalization [3]. Early and accurate risk stratification is therefore central to discharge planning, rehabilitation intensity, and advance-care discussions [4,5]. A quantitative model that estimates the expected recovery level at the time of ICU admission could help clinicians deploy resources to the patients who need them most.

Two complementary sources of information can inform such a model. First, the electronic health record, abbreviated EHR, provides a dense longitudinal stream of vital signs, neurological assessments, and laboratory values during the ICU stay, which reflects the acute physiological trajectory that static admission variables cannot capture [6,7]. Machine-learning models applied to such temporal features have generally outperformed static-only models for in-hospital mortality and clinical deterioration [8,9]. Second, stroke elicits a measurable systemic biological response. Peripheral blood transcriptomic studies have repeatedly documented coordinated remodeling of immune, inflammatory, and neural-repair pathways across the acute phase, which spans hours, and the subacute phase, which spans days to weeks [10,11]. Public gene-expression cohorts such as GSE37587 and GSE58294 therefore encode a temporal pathway atlas of stroke recovery that is independent of any single hospital's records [12,13].

Current approaches leave three gaps. Most EHR-based stroke outcome models treat the admission as a static snapshot and discard the temporal dynamics that may be most informative for recovery [14]. Deep sequential models that do consume ICU time series are typically difficult to interpret, and their training does not incorporate domain knowledge about the biological processes relevant to recovery [15,16]. Finally, although pathway-level priors have been used to regularize deep models in genomics [17,18], to our knowledge they have not been combined with EHR-derived clinical time series for ordinal functional-outcome prediction.

In this study, we propose an interpretable temporal pathway-regularized dynamic graph learning framework that addresses these gaps. We first constructed a MIMIC-IV stroke cohort of 4,858 admissions with a six-level ordinal discharge outcome and 2,875 patients with complete 64-step ICU time series and 218 static features. We then derived a temporal pathway atlas from four public stroke transcriptomic cohorts using single-sample gene-set enrichment, identifying 70 subacute-phase pathways with significant paired activity change in the derivation cohort. We treat this atlas as hypothesis-generating: it provides a mechanism-motivated prior for the clinical model, while its cross-cohort generalization is explicitly tested rather than assumed. We developed a dynamic graph gated recurrent unit regularized by this pathway prior, termed DG-GRU+TPR, and evaluated it within a stacked ensemble. Finally, we subjected the entire pipeline to a battery of robustness, negative-control, and leakage-audit analyses, so that the reported findings are accompanied by explicit evidence of their boundaries. The primary aim of this study was to deliver an ordinal stroke recovery model whose discrimination transfers across health systems and whose internal representation remains clinically interpretable; the secondary aim was to characterize precisely what the pathway prior contributes, rather than to claim an unqualified predictive increment. We hypothesized that encoding the prior as monotone-progression and bounded-drift constraints would reshape the latent recovery trajectory toward an interpretable, outcome-ordered progression while preserving discrimination, and that this structure, rather than a predictive gain, would constitute the prior's value.
