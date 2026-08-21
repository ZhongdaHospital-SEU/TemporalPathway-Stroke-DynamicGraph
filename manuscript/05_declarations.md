## 6. Declarations

### Data availability

The clinical data used in this study are from MIMIC-IV version 3.1 (https://physionet.org/content/mimiciv/) and the eICU Collaborative Research Database version 2.0 (https://physionet.org/content/eicu-crd/), publicly available de-identified critical-care databases distributed through PhysioNet, and from an independent cohort of 511 stroke patients with recorded 90-day modified Rankin Scale outcomes collected by the authors, and from a second independent cohort of 185 stroke patients with ICU time series and recorded 90-day modified Rankin Scale outcomes collected at a single tertiary hospital. Access to the PhysioNet databases requires credentialing and a data-use agreement; the derived cohort features used for modeling are available from the corresponding author upon reasonable request, subject to PhysioNet data-use restrictions and to the institutional data-use conditions of the two mRS cohorts. The transcriptomic data are publicly available from the Gene Expression Omnibus under accessions GSE37587, GSE58294, GSE16561, and GSE22255.

### Code availability

All analysis code (cohort extraction, temporal pathway atlas, model training and evaluation, and the robustness and leakage-audit pipelines) is available at https://github.com/wangzhipeng-1/TemporalPathway-Stroke-DynamicGraph. The repository is private during peer review; access will be granted to editors and reviewers on request, and the repository will be made public upon acceptance. A versioned archival copy with a digital object identifier will be deposited at the time of publication.

### Reporting guidelines

The study is reported in accordance with the Transparent Reporting of a multivariable prediction model for Individual Prognosis Or Diagnosis, TRIPOD, statement, and the Strengthening the Reporting of Observational Studies in Epidemiology, STROBE, checklist where applicable. The prediction model was developed and evaluated in external cohorts with prospective application in mind; the calibration and discrimination metrics reported follow the TRIPOD recommendations. A completed checklist is available from the corresponding author on request.

### Ethics approval and consent to participate

This study used de-identified data. Use of MIMIC-IV and the eICU-CRD was approved through credentialed PhysioNet access, and use of these public research databases was determined to be exempt from institutional review board approval. The independent mRS cohort was collected by the authors and approved by the Ethics Committee of Zhongda Hospital, Southeast University, China; all patients provided informed consent or were included under an approved waiver of consent, in accordance with applicable regulations. The hospital cohort of 185 patients was collected retrospectively from de-identified electronic health records at a single tertiary hospital in China and was approved by the Ethics Committee of Zhongda Hospital, Southeast University, China, with a waiver of informed consent. The GEO data were originally collected with participant consent and are redistributed in de-identified form.

### CRediT authorship contribution statement

Zhipeng Wang: Conceptualization, Methodology, Software, Formal analysis, Visualization, Writing – original draft. Luning Wang: Data curation, Investigation, Methodology, Writing – review & editing. Changsong Wang: Conceptualization, Supervision, Project administration, Funding acquisition, Writing – review & editing. Pengli Zhai: Data curation, Resources, Investigation. Hui Feng: Formal analysis, Visualization, Software. Hongmei Liu: Validation, Investigation, Resources. Qian Hou: Visualization, Software, Validation. Ming Guo: Validation, Writing – review & editing.

### Funding

This work was supported by the Higher Education Teaching Reform Research Project of Xuzhou Medical University (XYJG042).

### Competing interests

The authors declare that they have no competing interests.

### Use of artificial intelligence

During the preparation of this work the authors used artificial intelligence tools to support data analysis and manuscript drafting. After using these tools, the authors reviewed and edited the content as needed and take full responsibility for the content of the publication.

---

## 7. References

1. Martin SS, Aday AW, Almarzooq ZI, Anderson CA, Arora P, Avery CL et al. 2024 Heart Disease and Stroke Statistics: A Report of US and Global Data From the American Heart Association. Circulation 2024;149(8). doi:10.1161/CIR.0000000000001209

2. Feigin VL, Stark BA, Johnson CO, Roth GA, Bisignano C, Abady GG et al. Global, regional, and national burden of stroke and its risk factors, 1990–2019: a systematic analysis for the Global Burden of Disease Study 2019. The Lancet Neurology 2021;20(10):795-820. doi:10.1016/S1474-4422(21)00252-0

3. Abujaber A, Yaseen S, Imam Y, Nashwan A, Akhtar N. Machine learning-based prediction of one-year mortality in ischemic stroke patients. Oxford Open Neuroscience 2024;3. doi:10.1093/oons/kvae011

4. Chen Y, Ou Z, Deng Y, Luo A, Li X, Yang Y et al. Predictive Value of Machine Learning for Poststroke Mortality Risk: Systematic Review and Meta-Analysis. Journal of Medical Internet Research 2026;28:e83821. doi:10.2196/83821

5. Abedi V, Avula V, Razavi SM, Bavishi S, Chaudhary D, Shahjouei S et al. Predicting short and long-term mortality after acute ischemic stroke using EHR. Journal of the Neurological Sciences 2021;427:117560. doi:10.1016/j.jns.2021.117560

6. Ren W, Zhu J, Liu Z, Zhao T, Honavar V. A Comprehensive Survey of Electronic Health Record Modeling: From Deep Learning Approaches to Large Language Models. arXiv preprint arXiv:2507.12774 2025. doi:10.48550/arXiv.2507.12774

7. Le Baher H, Azé J, Bringay S, Poncelet P, Rodriguez N, Dunoyer C. Patient Electronic Health Record as Temporal Graphs for Health Monitoring. Studies in Health Technology and Informatics 2023. doi:10.3233/SHTI230205

8. Ashrafi N, Abdollahi A, Zhang J, Pishgar M. Optimizing Mortality Prediction for ICU Heart Failure Patients: Leveraging XGBoost and Advanced Machine Learning with the MIMIC-III Database. arXiv preprint arXiv:2409.01685 2024. doi:10.48550/arXiv.2409.01685

9. Iwagami M, Inokuchi R, Kawakami E, Yamada T, Goto A, Kuno T et al. Comparison of machine-learning and logistic regression models for prediction of 30-day unplanned readmission in electronic health records: A development and validation study. PLOS Digital Health 2024;3(8):e0000578. doi:10.1371/journal.pdig.0000578

10. Barr TL, VanGilder R, Rellick S, Brooks SD, Doll DN, Lucke-Wold AN et al. A Genomic Profile of the Immune Response to Stroke With Implications for Stroke Recovery. Biological Research For Nursing 2014;17(3):248-256. doi:10.1177/1099800414546492

11. Stamova B, Jickling GC, Ander BP, Zhan X, Liu D, Turner R et al. Gene Expression in Peripheral Immune Cells following Cardioembolic Stroke Is Sexually Dimorphic. PLoS ONE 2014;9(7):e102550. doi:10.1371/journal.pone.0102550

12. Barr T, Conley Y, Ding J, Dillman A, Warach S, Singleton A et al. Genomic biomarkers and cellular pathways of ischemic stroke by RNA gene expression profiling. Neurology 2010;75(11):1009-1014. doi:10.1212/WNL.0b013e3181f2b37f

13. Krug T, Gabriel JP, Taipa R, Fonseca BV, Domingues-Montanari S, Fernandez-Cadenas I et al. TTC7B Emerges as a Novel Risk Factor for Ischemic Stroke Through the Convergence of Several Genome-Wide Approaches. Journal of Cerebral Blood Flow & Metabolism 2012;32(6):1061-1072. doi:10.1038/jcbfm.2012.24

14. Arbet J, Brokamp C, Meinzen-Derr J, Trinkley KE, Spratt HM. Lessons and tips for designing a machine learning study using EHR data. Journal of Clinical and Translational Science 2021;5(1). doi:10.1017/cts.2020.513

15. Rossi E, Chamberlain B, Frasca F, Eynard D, Monti F, Bronstein M. Temporal Graph Networks for Deep Learning on Dynamic Graphs. arXiv preprint arXiv:2006.10637 2020. doi:10.48550/arXiv.2006.10637

16. Zhou J, Cui G, Hu S, Zhang Z, Yang C, Liu Z et al. Graph neural networks: A review of methods and applications. AI Open 2020;1:57-81. doi:10.1016/j.aiopen.2021.01.001

17. Hao J, Kim Y, Kim TK, Kang M. PASNet: pathway-associated sparse deep neural network for prognosis prediction from high-throughput data. BMC Bioinformatics 2018;19(1). doi:10.1186/s12859-018-2500-z

18. Deng L, Cai Y, Zhang W, Yang W, Gao B, Liu H. Pathway-Guided Deep Neural Network toward Interpretable and Predictive Modeling of Drug Sensitivity. Journal of Chemical Information and Modeling 2020;60(10):4497-4505. doi:10.1021/acs.jcim.0c00331

19. Johnson AEW, Bulgarelli L, Shen L, Gayles A, Shammout A, Horng S et al. MIMIC-IV, a freely accessible electronic health record dataset. Scientific Data 2023;10(1). doi:10.1038/s41597-022-01899-x

20. Johnson AEW, Pollard TJ, Shen L, Lehman LH, Feng M, Ghassemi M et al. MIMIC-III, a freely accessible critical care database. Scientific Data 2016;3:160035. doi:10.1038/sdata.2016.35

21. Pollard TJ, Johnson AEW, Raffa JD, Celi LA, Mark RG, Badawi O. The eICU Collaborative Research Database, a freely available multi-center database for critical care research. Scientific Data 2018;5:180178. doi:10.1038/sdata.2018.178

22. Collins GS, Moons KGM, Dhiman P, Riley RD, Beam AL, Van Calster B et al. TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods. BMJ 2024;e078378. doi:10.1136/bmj-2023-078378

23. Mickle CF, Deb D. Early prediction of patient discharge disposition in acute neurological care using machine learning. BMC Health Services Research 2022;22(1). doi:10.1186/s12913-022-08615-w

24. Vafaei Sadr A, Li J, Hwang W, Yeasin M, Wang M, Lehmann H et al. Flexible imputation toolkit for electronic health records. Scientific Reports 2025;15(1). doi:10.1038/s41598-025-02276-5

25. Hänzelmann S, Castelo R, Guinney J. GSVA: gene set variation analysis for microarray and RNA-Seq data. BMC Bioinformatics 2013;14(1). doi:10.1186/1471-2105-14-7

26. Barbie DA, Tamayo P, Boehm JS, Kim SY, Moody SE, Dunn IF et al. Systematic RNA interference reveals that oncogenic KRAS-driven cancers require TBK1. Nature 2009;462(7269):108-112. doi:10.1038/nature08460

27. Fabregat A, Jupe S, Matthews L, Sidiropoulos K, Gillespie M, Garapati P et al. The Reactome Pathway Knowledgebase. Nucleic Acids Research 2018;46(D1):D649-D655. doi:10.1093/nar/gkx1132

28. Sales G, Calura E, Cavalieri D, Romualdi C. graphite - a Bioconductor package to convert pathway topology to gene network. BMC Bioinformatics 2012;13(1). doi:10.1186/1471-2105-13-20

29. Martini P, Sales G, Massa MS, Chiogna M, Romualdi C. Along signal paths: an empirical gene set approach exploiting pathway topology. Nucleic Acids Research 2013;41(1):e19-e19. doi:10.1093/nar/gks866

30. Muzio G, O’Bray L, Borgwardt K. Biological network analysis with deep learning. Briefings in Bioinformatics 2021;22(2):1515-1530. doi:10.1093/bib/bbaa257

31. Oh JH, Choi W, Ko E, Kang M, Tannenbaum A, Deasy JO. PathCNN: interpretable convolutional neural networks for survival prediction and pathway analysis applied to glioblastoma. Bioinformatics 2021;37(Supplement_1):i443-i450. doi:10.1093/bioinformatics/btab285

32. Zhao L, Dong Q, Luo C, Wu Y, Bu D, Qi X et al. DeepOmix: A scalable and interpretable multi-omics deep learning framework and application in cancer survival analysis. Computational and Structural Biotechnology Journal 2021;19:2719-2725. doi:10.1016/j.csbj.2021.04.067

33. Mallavarapu T, Hao J, Kim Y, Oh JH, Kang M. Pathway-based deep clustering for molecular subtyping of cancer. Methods 2020;173:24-31. doi:10.1016/j.ymeth.2019.06.017

34. Chen T, Guestrin C. XGBoost. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining 2016;785-794. doi:10.1145/2939672.2939785

35. Cho K, van Merrienboer B, Gulcehre C, Bahdanau D, Bougares F, Schwenk H et al. Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation. arXiv preprint arXiv:1406.1078 2014. doi:10.48550/arXiv.1406.1078

36. Che Z, Purushotham S, Cho K, Sontag D, Liu Y. Recurrent Neural Networks for Multivariate Time Series with Missing Values. Scientific Reports 2018;8(1):6085. doi:10.1038/s41598-018-24271-9

37. Vaswani A, Shazeer N, Parmar N, Uszkoreit J, Jones L, Gomez AN et al. Attention Is All You Need. Advances in Neural Information Processing Systems 2017;30:5998-6008. doi:10.48550/arXiv.1706.03762

38. Yu B, Yin H, Zhu Z. Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting. Proceedings of the Twenty-Seventh International Joint Conference on Artificial Intelligence 2018;3634-3640. doi:10.24963/ijcai.2018/153

39. Scarselli F, Gori M, Ah Chung Tsoi, Hagenbuchner M, Monfardini G. The Graph Neural Network Model. IEEE Transactions on Neural Networks 2009;20(1):61-80. doi:10.1109/tnn.2008.2005605

40. Pareja A, Domeniconi G, Chen J, Ma T, Suzumura T, Kanezashi H et al. EvolveGCN: Evolving Graph Convolutional Networks for Dynamic Graphs. Proceedings of the AAAI Conference on Artificial Intelligence 2020;34(4):5363-5370. doi:10.1609/aaai.v34i04.5984

41. Hu Z, Dong Y, Wang K, Sun Y. Heterogeneous Graph Transformer. Proceedings of The Web Conference 2020 2020;2704-2710. doi:10.1145/3366423.3380027

42. Zhang C, Song D, Huang C, Swami A, Chawla NV. Heterogeneous Graph Neural Network. Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining 2019;793-803. doi:10.1145/3292500.3330961

43. Dong Y, Chawla NV, Swami A. metapath2vec. Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining 2017;135-144. doi:10.1145/3097983.3098036

44. Grover A, Leskovec J. node2vec. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining 2016;855-864. doi:10.1145/2939672.2939754

45. Chen Y, Tang X, Qi X, Li CG, Xiao R. Learning graph normalization for graph neural networks. Neurocomputing 2022;493:613-625. doi:10.1016/j.neucom.2022.01.003

46. Paszke A, Gross S, Massa F, Lerer A, Bradbury J, Chanan G et al. PyTorch: An Imperative Style, High-Performance Deep Learning Library. arXiv preprint arXiv:1912.01703 2019. doi:10.48550/arXiv.1912.01703

47. Pedregosa F, Varoquaux G, Gramfort A, Michel V, Thirion B, Grisel O et al. Scikit-learn: Machine Learning in Python. arXiv preprint arXiv:1201.0490 2012. doi:10.48550/arXiv.1201.0490

48. Piya FL, Gupta M, Beheshti R. HealthGAT: Node Classifications in Electronic Health Records using Graph Attention Networks. 2024 IEEE/ACM Conference on Connected Health: Applications, Systems and Engineering Technologies (CHASE) 2024;132-141. doi:10.1109/chase60773.2024.00022

49. Oss Boll H, Amirahmadi A, Ghazani MM, Morais WOD, Freitas EPD, Soliman A et al. Graph neural networks for clinical risk prediction based on electronic health records: A survey. Journal of Biomedical Informatics 2024;151:104616. doi:10.1016/j.jbi.2024.104616

50. Yaseliani M, Noor-E-Alam M, Dasa O, Xian X, Pepine CJ, Hasan MM. A lightweight graph neural network to predict long-term mortality in coronary artery disease patients: an interpretable causality-aware approach. Journal of Biomedical Informatics 2025;167:104846. doi:10.1016/j.jbi.2025.104846

51. Jia Z, Zeng X, Duan H, Lu X, Li H. A patient-similarity-based model for diagnostic prediction. International Journal of Medical Informatics 2020;135:104073. doi:10.1016/j.ijmedinf.2019.104073

52. Sahu MK, Roy P. Similarity-Based Self-Construct Graph Model for Predicting Patient Criticalness Using Graph Neural Networks and EHR Data. arXiv preprint arXiv:2508.00615 2025. doi:10.48550/arXiv.2508.00615

53. Lin KW, Kuo YC, Wang HY, Tseng YJ. KAT-GNN: A Knowledge-Augmented Temporal Graph Neural Network for Risk Prediction in Electronic Health Records. arXiv preprint arXiv:2511.01249 2025. doi:10.48550/arXiv.2511.01249

54. Cohen J. A Coefficient of Agreement for Nominal Scales. Educational and Psychological Measurement 1960;20(1):37-46. doi:10.1177/001316446002000104

55. DeLong ER, DeLong DM, Clarke-Pearson DL. Comparing the Areas under Two or More Correlated Receiver Operating Characteristic Curves: A Nonparametric Approach. Biometrics 1988;44(3):837. doi:10.2307/2531595

56. Pencina MJ, D' Agostino RB, D' Agostino RB, Vasan RS. Evaluating the added predictive ability of a new marker: From area under the ROC curve to reclassification and beyond. Statistics in Medicine 2008;27(2):157-172. doi:10.1002/sim.2929

57. Ruopp MD, Perkins NJ, Whitcomb BW, Schisterman EF. Youden Index and Optimal Cut‐Point Estimated from Observations Affected by a Lower Limit of Detection. Biometrical Journal 2008;50(3):419-430. doi:10.1002/bimj.200710415

58. Vickers AJ, Elkin EB. Decision Curve Analysis: A Novel Method for Evaluating Prediction Models. Medical Decision Making 2006;26(6):565-574. doi:10.1177/0272989X06295361

59. Lundberg S, Lee SI. A Unified Approach to Interpreting Model Predictions. arXiv preprint arXiv:1705.07874 2017. doi:10.48550/arXiv.1705.07874

60. Tharzeen A, Sadr AV, Radfar N, Hwang W, Abedi V, Zand R. A Heterogeneous Graph Neural Network Framework for Multi-Horizon Stroke Mortality Prediction. medRxiv [preprint] 2026. doi:10.64898/2026.06.09.26355176

61. Liang B, Gong H, Lu L, Xu J. Risk stratification and pathway analysis based on graph neural network and interpretable algorithm. BMC Bioinformatics 2022;23(1). doi:10.1186/s12859-022-04950-1

62. Cummins JA, Gerber BS, Fukunaga MI, Henninger N, Kiefe CI, Liu F. In-Hospital Mortality Prediction among Intensive Care Unit Patients with Acute Ischemic Stroke: A Machine Learning Approach. Health Data Science 2025;5. doi:10.34133/hds.0179

63. Georgiev K, Doudesis D, McPeake J, Mills NL, Shenkin SD, Fleuriot JD et al. Machine learning-based predictions of healthcare contacts following emergency hospitalisation using electronic health records. npj Digital Medicine 2025;8(1). doi:10.1038/s41746-025-02138-4
