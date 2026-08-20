# -*- coding: utf-8 -*-
"""Nested-CV stacking of clinical XGBoost + temporal pathway-regularized GRU.

Level-0 models (5-fold OOF, same split): XGB static+TS, DG-GRU+TPR, (optionally
plain GRU). Level-1: logistic regression on concatenated OOF probabilities,
fitted per fold on the other 4 folds to avoid leakage.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, cohen_kappa_score, roc_auc_score
import warnings
warnings.filterwarnings("ignore")

RESULTS = r"D:\TT paper\0811Temporal Pathway\results"
N_CLASSES = 6


def multiclass_auc(y_true, proba):
    aucs = []
    for k in range(N_CLASSES):
        try:
            aucs.append(roc_auc_score((y_true == k).astype(int), proba[:, k]))
        except Exception:
            pass
    return float(np.mean(aucs)) if aucs else 0.0


def bootstrap_auc_diff(y, p1, p2, n_boot=1000, seed=7):
    """Bootstrap CI for macro-AUC(p1) - macro-AUC(p2) on paired samples."""
    rng = np.random.default_rng(seed)
    n = len(y)
    d = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        a1 = multiclass_auc(y[idx], p1[idx])
        a2 = multiclass_auc(y[idx], p2[idx])
        d.append(a1 - a2)
    d = np.array(d)
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main():
    cv = np.load(os.path.join(RESULTS, "cv_predictions.npz"))
    y = cv["y_icu"]
    xgb_s = cv["xgb_icu_static"]
    xgb_ts = cv["xgb_icu_ts"]
    dg = np.load(os.path.join(RESULTS, "dggru_tpr_oof.npz"))
    assert np.array_equal(y, dg["y"])
    p_dg = dg["proba"]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(skf.split(np.zeros(len(y)), y))

    def stack(feature_sets):
        """feature_sets: list of (name, proba). Nested CV stack."""
        oof = np.zeros((len(y), N_CLASSES))
        for tr, te in splits:
            Xtr = np.concatenate([p[tr] for _, p in feature_sets], axis=1)
            Xte = np.concatenate([p[te] for _, p in feature_sets], axis=1)
            clf = LogisticRegression(max_iter=1000, C=0.1)
            clf.fit(Xtr, y[tr])
            oof[te] = clf.predict_proba(Xte)
        return oof

    combos = {
        "stack_xgb_ts+dggru_tpr": [("xgb_ts", xgb_ts), ("dggru_tpr", p_dg)],
        "stack_xgb_ts+xgb_s+dggru_tpr": [("xgb_ts", xgb_ts), ("xgb_s", xgb_s), ("dggru_tpr", p_dg)],
    }
    lines = []
    models = {"XGB static+TS": xgb_ts, "DG-GRU+TPR": p_dg}
    for name, feats in combos.items():
        models[name] = stack(feats)

    print("=== Stacking results (nested 5-fold CV) ===", flush=True)
    for name, p in models.items():
        pred = p.argmax(1)
        line = (f"{name:28s} acc={accuracy_score(y, pred):.4f} "
                f"kappa={cohen_kappa_score(y, pred):.4f} "
                f"macroAUC={multiclass_auc(y, p):.4f}")
        print(line, flush=True)
        lines.append(line)
    # bootstrap significance vs XGB static+TS
    for name in ["DG-GRU+TPR", "stack_xgb_ts+dggru_tpr", "stack_xgb_ts+xgb_s+dggru_tpr"]:
        lo, hi = bootstrap_auc_diff(y, models[name], xgb_ts)
        print(f"{name:28s} vs XGB static+TS: AUC diff [{lo:+.4f}, {hi:+.4f}]", flush=True)
        lines.append(f"{name} vs XGB static+TS: AUC diff 95%CI [{lo:+.4f}, {hi:+.4f}]")
    np.savez(os.path.join(RESULTS, "stack_oof.npz"),
             y=y, xgb_ts=xgb_ts, dggru_tpr=p_dg,
             stack1=models["stack_xgb_ts+dggru_tpr"],
             stack2=models["stack_xgb_ts+xgb_s+dggru_tpr"],
             proba=models["stack_xgb_ts+xgb_s+dggru_tpr"])
    with open(os.path.join(RESULTS, "stacking_result.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
