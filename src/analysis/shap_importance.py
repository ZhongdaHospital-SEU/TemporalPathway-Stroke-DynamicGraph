# -*- coding: utf-8 -*-
"""SHAP feature importance for the XGB static+TS model (clinical baseline)."""
from __future__ import annotations
import os
import sys
import numpy as np
import pandas as pd
import xgboost as xgb
# shap incompatible with xgboost 3.x multi-class; using gain importance
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROC = r"D:\TT paper\0811Temporal Pathway\data\processed\mimic_stroke"
RESULTS = r"D:\TT paper\0811Temporal Pathway\results"
FIG = os.path.join(RESULTS, "figures")
os.makedirs(FIG, exist_ok=True)
N_CLASSES = 6


def main():
    stat = pd.read_csv(os.path.join(PROC, "static_plus_ts.csv"))
    stat = stat[stat["outcome_ordinal"].notna()].copy()
    z = np.load(os.path.join(PROC, "icu_sequences.npz"))
    seq_hadms = set(int(k[4:]) for k in z.files if k.startswith("seq_"))
    stat = stat[stat["hadm_id"].isin(seq_hadms)].copy()
    drop = ["hadm_id", "subject_id", "discharge_location", "outcome_ordinal", "hospital_expire_flag"]
    X = stat.drop(columns=[c for c in drop if c in stat.columns])
    X = X.select_dtypes(include=[np.number])
    X = X.loc[:, X.median() == X.median()]
    X = X.fillna(X.median())
    y = stat["outcome_ordinal"].astype(int).values
    model = xgb.XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8,
                              objective="multi:softprob", num_class=N_CLASSES,
                              tree_method="hist", random_state=42, n_jobs=4)
    model.fit(X, y)
    # xgboost 3.x + shap incompatibility for multi:softprob -> use gain importance
    imp = model.feature_importances_  # gain-based
    order = np.argsort(imp)[::-1][:20]
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(range(20), imp[order][::-1], color="#4C72B0")
    ax.set_yticks(range(20))
    ax.set_yticklabels([str(X.columns[i]) for i in order[::-1]], fontsize=8)
    ax.set_xlabel("XGB gain importance")
    ax.set_title("Top-20 features, XGB static+TS")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "shap_importance.png"), dpi=200)
    plt.close()
    top = pd.DataFrame({"feature": [str(X.columns[i]) for i in order], "gain": imp[order]})
    top.to_csv(os.path.join(RESULTS, "shap_top20.csv"), index=False)
    print("[shap] saved shap_importance.png + shap_top20.csv (gain-based)", flush=True)
    print(top.head(12).to_string(), flush=True)


if __name__ == "__main__":
    main()
