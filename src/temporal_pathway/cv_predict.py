# -*- coding: utf-8 -*-
"""Cross-validated out-of-fold predictions for baseline models (XGBoost).

Same 5-fold split as the GRU model (StratifiedKFold seed 42) so NRI/IDI/DCA
comparisons are paired. Saves OOF probabilities to results/cv_predictions.npz.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, cohen_kappa_score, roc_auc_score
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

PROC = r"D:\TT paper\0811Temporal Pathway\data\processed\mimic_stroke"
RESULTS = r"D:\TT paper\0811Temporal Pathway\results"
OUT = os.path.join(RESULTS, "cv_predictions.npz")

N_CLASSES = 6


def load_data():
    stat = pd.read_csv(os.path.join(PROC, "static_features.csv"))
    ts = pd.read_csv(os.path.join(PROC, "static_plus_ts.csv"))
    stat = stat[stat["outcome_ordinal"].notna()].copy()
    ts = ts[ts["outcome_ordinal"].notna()].copy()
    # ICU subset = patients with sequences
    z = np.load(os.path.join(PROC, "icu_sequences.npz"))
    seq_hadms = set(int(k[4:]) for k in z.files if k.startswith("seq_"))
    drop = ["hadm_id", "subject_id", "discharge_location", "outcome_ordinal", "hospital_expire_flag"]
    y_full = stat["outcome_ordinal"].astype(int).values
    X_full = stat.drop(columns=[c for c in drop if c in stat.columns])
    X_full = X_full.select_dtypes(include=[np.number])
    X_full = X_full.loc[:, X_full.median() == X_full.median()]
    X_full = X_full.fillna(X_full.median())

    y_icu = ts["outcome_ordinal"].astype(int).values
    X_icu = ts.drop(columns=[c for c in drop if c in ts.columns])
    X_icu = X_icu.select_dtypes(include=[np.number])
    X_icu = X_icu.loc[:, X_icu.median() == X_icu.median()]
    X_icu = X_icu.fillna(X_icu.median())
    # keep only ICU-subset rows in ts
    hadms_ts = ts["hadm_id"].values
    keep = np.array([h in seq_hadms for h in hadms_ts])
    y_icu = y_icu[keep]
    X_icu = X_icu.iloc[keep]
    return X_full, y_full, X_icu, y_icu


def multiclass_auc(y_true, proba):
    aucs = []
    for k in range(N_CLASSES):
        try:
            aucs.append(roc_auc_score((y_true == k).astype(int), proba[:, k]))
        except Exception:
            pass
    return float(np.mean(aucs)) if aucs else 0.0


def xgb_cv(X, y, label, name):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    proba = np.zeros((len(y), N_CLASSES))
    accs, kaps, aucs = [], [], []
    for tr, te in skf.split(X, y):
        sc = StandardScaler().fit(X.iloc[tr])
        Xtr = sc.transform(X.iloc[tr]); Xte = sc.transform(X.iloc[te])
        model = xgb.XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                                  subsample=0.8, colsample_bytree=0.8,
                                  objective="multi:softprob", num_class=N_CLASSES,
                                  tree_method="hist", random_state=42,
                                  n_jobs=4, eval_metric="mlogloss")
        model.fit(Xtr, y[tr], verbose=False)
        p = model.predict_proba(Xte)
        proba[te] = p
        pred = p.argmax(1)
        accs.append(accuracy_score(y[te], pred))
        kaps.append(cohen_kappa_score(y[te], pred))
        aucs.append(multiclass_auc(y[te], p))
    line = (f"XGB {label} | acc: {np.mean(accs):.4f} | kappa: {np.mean(kaps):.4f} | "
            f"auc: {np.mean(aucs):.4f}")
    print(line, flush=True)
    return proba, line


def main():
    X_full, y_full, X_icu, y_icu = load_data()
    print(f"[cv] full cohort n={len(y_full)}, icu subset n={len(y_icu)}", flush=True)
    # full-cohort static baseline
    p_full, l1 = xgb_cv(X_full, y_full, "static-only (full cohort)", "xgb_full")
    # ICU-subset baselines
    static_cols = set(pd.read_csv(os.path.join(PROC, "static_features.csv"), nrows=0).columns)
    X_icu_static = X_icu[[c for c in X_icu.columns if c in static_cols]]
    p_icu_static, l2 = xgb_cv(X_icu_static, y_icu, "static-only (ICU subset)", "xgb_icu_static")
    p_icu_ts, l3 = xgb_cv(X_icu, y_icu, "static+TS (ICU subset)", "xgb_icu_ts")
    np.savez(OUT,
             y_full=y_full, xgb_full=p_full,
             y_icu=y_icu, xgb_icu_static=p_icu_static, xgb_icu_ts=p_icu_ts)
    with open(os.path.join(RESULTS, "cv_baselines.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join([l1, l2, l3]) + "\n")


if __name__ == "__main__":
    main()
