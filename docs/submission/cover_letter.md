# Cover Letter (draft)

**To:** The Editors, Computational and Structural Biotechnology Journal

**Re:** Submission of original research article: "Temporal Pathway-Regularized Dynamic Graph Learning for Ordinal Stroke Recovery Prediction from Intensive Care Time Series"

Dear Editor,

We are pleased to submit our original research article for consideration as a research article in Computational and Structural Biotechnology Journal.

Stroke recovery is heterogeneous, and discharge planning depends on early, accurate estimates of the expected functional outcome. Electronic health records contain dense ICU time series that capture the acute physiological trajectory, while public blood transcriptomic cohorts encode a temporal atlas of stroke-related pathway remodeling. These two data modalities have not previously been combined in a single predictive framework for ordinal recovery outcomes.

We integrate them in a temporal pathway-regularized dynamic graph learning framework. A 64-step hourly ICU sequence of physiological and neurological channels feeds a dynamic graph gated recurrent unit, whose latent recovery trajectory is regularized by a 70-pathway temporal atlas derived from four public stroke transcriptomic cohorts. The model predicts a six-level ordinal discharge outcome in 2,875 MIMIC-IV stroke admissions. The stacked ensemble reached a macro-AUC of 0.830. Ablation, negative-control, regularization-grid, seed, label, and information-leakage analyses characterized precisely what the pathway prior contributes, namely interpretable structure rather than a predictive gain. External validation in 12,820 eICU-CRD admissions preserved discrimination, with a four-class macro-AUC of 0.809 for direct transfer and a mortality AUROC of 0.962, and results were stable under a narrow ischemic-stroke cohort definition.

We believe the work fits the scope of Computational and Structural Biotechnology Journal because it addresses a biomedical prediction problem with a computational method that combines temporal deep learning, biological pathway priors, and rigorous external validation, and it reports both the strengths and the boundaries of the method.

This manuscript is original, has not been published previously, and is not under consideration for publication elsewhere. All authors have read and approved the final manuscript and agree to its submission. The authors declare no competing interests. Data sources used are publicly available under credentialed access; details are provided in the Data Availability statement.

Thank you for your consideration. We look forward to your response.

Sincerely,
[Corresponding Author Name], on behalf of all authors
[Affiliation]
[Email]
[Date]
