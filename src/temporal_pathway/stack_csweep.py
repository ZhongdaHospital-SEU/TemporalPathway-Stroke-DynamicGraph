# -*- coding: utf-8 -*-
"""Stacker hyperparameter sweep (logistic-regression C) on saved OOFs."""
from __future__ import annotations
import os
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, cohen_kappa_score, roc_auc_score

RESULTS = r"D:\TT paper\0811Temporal Pathway\results"
N_CLASSES = 6


def multiclass_auc(y, p):
    aucs = []
    for k in range(N_CLASSES):
        try:
            aucs.append(roc_auc_score((y == k).astype(int), p[:, k]))
        except Exception:
            pass
    return float(np.mean(aucs)) if aucs else 0.0


def main():
    st = np.load(os.path.join(RESULTS, "stack_oof.npz"))
    y = st["y"]
    feats = [st["xgb_ts"], st["xgb_icu_static"] if "xgb_icu_static" in st.files else st["xgb_ts"], st["dggru_tpr"]]
    # use only xgb_ts + dggru_tpr + xgb_static from cv_predictions
    cv = np.load(os.path.join(RESULTS, "cv_predictions.npz"))
    X = np.concatenate([st["xgb_ts"], cv["xgb_icu_static"], st["dggru_tpr"]], axis=1)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(skf.split(np.zeros(len(y)), y))
    lines = []
    for C in [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
        oof = np.zeros((len(y), N_CLASSES))
        for tr, te in splits:
            clf = LogisticRegression(max_iter=2000, C=C)
            clf.fit(X[tr], y[tr])
            oof[te] = clf.predict_proba(X[te])
        pred = oof.argmax(1)
        line = (f"C={C:<5} acc={accuracy_score(y, pred):.4f} "
                f"kappa={cohen_kappa_score(y, pred):.4f} auc={multiclass_auc(y, oof):.4f}")
        print(line, flush=True)
        lines.append(line)
    with open(os.path.join(RESULTS, "stack_csweep.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
