# 结果数字主库（写作用唯一真相源，2026-08-14 生成）

> 数值统一 2 位小数；P 值规则：P≥0.001 保留 3 位，P<0.001 一律写 P<0.001。所有数字源自 results/ 下文件，若与正文出入以本文件为准。

## 模型对比（5 折分层 OOF，ICU 子集 n=2875）
| 模型 | acc | kappa | macro-AUC | Somers'D | Brier |
|---|---|---|---|---|---|
| XGB static | 0.51 | 0.38 | 0.82 | 0.64 | 0.10 |
| XGB static+TS | 0.53 | 0.40 | 0.83 | 0.64 | 0.10 |
| dggru | 0.48 | 0.37 | 0.81 | 0.63 | 0.11 |
| dggru_tpr | 0.47 | 0.36 | 0.81 | 0.63 | 0.11 |
| gru_long | 0.46 | 0.35 | 0.80 | 0.62 | 0.11 |
| gru | 0.47 | 0.37 | 0.81 | 0.63 | 0.11 |
| gru_seqonly | 0.42 | 0.30 | 0.78 | 0.57 | 0.11 |
| gru_tpr | 0.47 | 0.37 | 0.81 | 0.63 | 0.11 |
| gru_tsstatic | 0.48 | 0.37 | 0.81 | 0.63 | 0.11 |
| v2_bi_tsstatic | 0.47 | 0.36 | 0.80 | 0.61 | 0.12 |
| v2_dggru_tpr_bi_tsstatic | 0.47 | 0.36 | 0.80 | 0.61 | 0.12 |
| Stack (proposed) | 0.54 | 0.41 | 0.83 | 0.65 | 0.10 |

## 增量价值（NRI/IDI）
### icu_timeseries（mean NRI=0.19）
| cutoff | NRI | IDI |
|---|---|---|
| y>=1 | 0.10 | 0.00 |
| y>=2 | 0.10 | 0.00 |
| y>=3 | 0.13 | 0.01 |
| y>=4 | 0.18 | 0.02 |
| y>=5 | 0.46 | 0.06 |

### pathway_prior（mean NRI=-0.28）
| cutoff | NRI | IDI |
|---|---|---|
| y>=1 | -0.25 | -0.00 |
| y>=2 | -0.20 | -0.00 |
| y>=3 | -0.34 | -0.00 |
| y>=4 | -0.36 | -0.01 |
| y>=5 | -0.23 | -0.01 |

## DeLong（Stack vs XGB static+TS，per class）
| class | dAUC | z | p |
|---|---|---|---|
| 1 | -0.00 | -0.31 | P=0.755 |

## DCA 净获益（XGB static+TS vs Stack）
| cutoff | pt | NB XGB+TS | NB Stack | Δ |
|---|---|---|---|---|
| y>=2 | 0.10 | 0.68 | 0.68 | 0.00 |
| y>=2 | 0.20 | 0.64 | 0.64 | -0.00 |
| y>=2 | 0.30 | 0.61 | 0.61 | -0.00 |
| y>=3 | 0.10 | 0.37 | 0.37 | -0.01 |
| y>=3 | 0.20 | 0.33 | 0.33 | -0.00 |
| y>=3 | 0.30 | 0.28 | 0.29 | 0.01 |
| y>=5 | 0.10 | 0.14 | 0.15 | 0.00 |
| y>=5 | 0.20 | 0.14 | 0.14 | 0.00 |
| y>=5 | 0.30 | 0.13 | 0.13 | -0.00 |

## stacking_result
```
XGB static+TS                acc=0.5297 kappa=0.4024 macroAUC=0.8260
DG-GRU+TPR                   acc=0.4699 kappa=0.3643 macroAUC=0.8141
stack_xgb_ts+dggru_tpr       acc=0.5360 kappa=0.4009 macroAUC=0.8295
stack_xgb_ts+xgb_s+dggru_tpr acc=0.5391 kappa=0.4073 macroAUC=0.8304
DG-GRU+TPR vs XGB static+TS: AUC diff 95%CI [-0.0188, -0.0050]
stack_xgb_ts+dggru_tpr vs XGB static+TS: AUC diff 95%CI [-0.0009, +0.0077]
stack_xgb_ts+xgb_s+dggru_tpr vs XGB static+TS: AUC diff 95%CI [+0.0005, +0.0084]
```

## ts_increment
```
XGB static-only | acc: 0.4835+/-0.0114 | kappa: 0.3585 | auc: 0.8100+/-0.0094
XGB static+TS | acc: 0.4928+/-0.0092 | kappa: 0.3700 | auc: 0.8134+/-0.0118
```

## cv_baselines
```
XGB static-only (full cohort) | acc: 0.4901 | kappa: 0.3663 | auc: 0.8142
XGB static-only (ICU subset) | acc: 0.5117 | kappa: 0.3809 | auc: 0.8175
XGB static+TS (ICU subset) | acc: 0.5297 | kappa: 0.4024 | auc: 0.8262
```

## feasibility_baseline
```
LogReg(OVR) | acc: 0.4619 +/- 0.0111
LogReg(OVR) | kappa: 0.3333 +/- 0.0138
LogReg(OVR) | auc: 0.7850 +/- 0.0062
RandomForest | acc: 0.4716 +/- 0.0137
RandomForest | kappa: 0.3338 +/- 0.0172
RandomForest | auc: 0.7983 +/- 0.0117
XGBoost | acc: 0.4835 +/- 0.0114
XGBoost | kappa: 0.3585 +/- 0.0139
XGBoost | auc: 0.8100 +/- 0.0094
Majority-class baseline accuracy: 0.2476
```

## 表数据（Table1-5，CSV 原文）
### Table1
```
Outcome	n	Age (mean)	Male (%)	ICU-seq (%)	LOS (h, median)	Obs (median)	First GCS (median)
HOME	1088	58.85	56.7%	43.0%	49.13	61	15
HOME CARE	832	68.61	49.0%	43.9%	68.28	78	15
REHAB	1203	67.58	50.5%	69.2%	116.00	122	14
SNF	761	74.00	43.0%	51.8%	94.64	113	13
HOSPICE	424	71.74	46.5%	77.6%	192.73	201	11
DIED	550	70.90	51.8%	88.4%	140.85	159	8
```

### Table2
```
Model	acc	kappa	macro-AUC
XGB static-only (full cohort)	0.49	0.37	0.814
XGB static-only (ICU subset)	0.51	0.38	0.818
XGB static+TS (ICU subset)	0.53	0.40	0.826
GRU seq-only	0.42	0.30	0.777
GRU+static	0.47	0.37	0.814
GRU+TPR	0.47	0.37	0.815
GRU+static+TS	0.48	0.37	0.812
DG-GRU	0.48	0.37	0.814
DG-GRU+TPR	0.47	0.36	0.815
GRU long (60ep)	0.46	0.35	0.804
Stack (proposed)	0.54	0.41	0.830
```

### Table3
```
Model	AUC	dAUC(GRU+TPR)	dAUC(DG-GRU+TPR)
GRU	0.813	0.002	nan
GRU+TPR	0.815	-	nan
DG-GRU	0.813	nan	0.001
DG-GRU+TPR	0.814	nan	-
```

### Table4
```
Increment	Cutoff	NRI	IDI
ICU timeseries (XGB static+TS vs XGB static)	y>=1	0.100	0.002
ICU timeseries (XGB static+TS vs XGB static)	y>=2	0.095	0.003
ICU timeseries (XGB static+TS vs XGB static)	y>=3	0.131	0.007
ICU timeseries (XGB static+TS vs XGB static)	y>=4	0.177	0.016
ICU timeseries (XGB static+TS vs XGB static)	y>=5	0.460	0.060
pathway prior (DG-GRU+TPR vs DG-GRU)	y>=1	-0.255	-0.002
pathway prior (DG-GRU+TPR vs DG-GRU)	y>=2	-0.195	-0.000
pathway prior (DG-GRU+TPR vs DG-GRU)	y>=3	-0.345	-0.004
pathway prior (DG-GRU+TPR vs DG-GRU)	y>=4	-0.363	-0.006
pathway prior (DG-GRU+TPR vs DG-GRU)	y>=5	-0.232	-0.012
```

### Table5
```
Feature group	Gain
Labs	0.48
GCS	0.19
Vitals	0.15
Motor exam	0.09
Other	0.07
Demographics	0.01
```
