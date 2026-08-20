# -*- coding: utf-8 -*-
"""Clinical utility figures: DCA, calibration, confusion matrix, per-class ROC."""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc as roc_auc, confusion_matrix, ConfusionMatrixDisplay

RESULTS = r"D:\TT paper\0811Temporal Pathway\results"
FIG = os.path.join(RESULTS, "figures")
os.makedirs(FIG, exist_ok=True)
LABELS = ["HOME", "HOME CARE", "REHAB", "SNF", "HOSPICE", "DIED"]


def fig_dca():
    d = pd.read_csv(os.path.join(RESULTS, "dca_curves.csv"))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for ax, k in zip(axes, [2, 3, 5]):
        sub = d[d["cutoff"] == k]
        ax.plot(sub["pt"], sub["xgb_static"], "b-", lw=1.8, label="XGB static+TS")
        ax.plot(sub["pt"], sub["proposed"], "r-", lw=1.8, label="Stack (proposed)")
        ax.plot(sub["pt"], sub["treat_all"], "k--", lw=1, label="Treat all")
        ax.plot(sub["pt"], np.zeros_like(sub["pt"]), "k:", lw=1, label="Treat none")
        ax.set_title(f"Outcome y>={k}")
        ax.set_xlabel("Threshold probability")
        ax.set_ylim(-0.05, 0.75)
        if k == 2:
            ax.set_ylabel("Net benefit")
            ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "dca_curves.png"), dpi=200)
    plt.close()
    print("saved dca_curves.png")


def fig_calibration():
    d = pd.read_csv(os.path.join(RESULTS, "calibration.csv"))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True, sharey=True)
    for ax, k in zip(axes, [1, 2, 3]):
        for name, color in [("xgb_static+ts", "b"), ("Stack (proposed)", "r")]:
            sub = d[(d["cutoff"] == k) & (d["model"] == name)]
            ax.plot([0, 1], [0, 1], "k--", lw=1)
            ax.plot(sub["pred"], sub["obs"], "o-", color=color, ms=4, label=name)
        ax.set_title(f"P(y>={k})")
        ax.set_xlabel("Predicted"); ax.set_ylabel("Observed")
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "calibration.png"), dpi=200)
    plt.close()
    print("saved calibration.png")


def fig_cm():
    st = np.load(os.path.join(RESULTS, "stack_oof.npz"))
    y = st["y"]
    pred = st["proba"].argmax(1)
    cm = confusion_matrix(y, pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=LABELS)
    fig, ax = plt.subplots(figsize=(8.5, 7))
    disp.plot(ax=ax, cmap="Blues", colorbar=True)
    ax.set_title("Stack (proposed) confusion matrix (5-fold OOF)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "confusion_matrix.png"), dpi=200)
    plt.close()
    print("saved confusion_matrix.png")


def fig_roc():
    st = np.load(os.path.join(RESULTS, "stack_oof.npz"))
    cv = np.load(os.path.join(RESULTS, "cv_predictions.npz"))
    y = cv["y_icu"]
    fig, ax = plt.subplots(figsize=(7.5, 6))
    for k in range(6):
        yb = (y == k).astype(int)
        fpr, tpr, _ = roc_curve(yb, st["proba"][:, k])
        a = roc_auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=1.6, label=f"{LABELS[k]} (AUC={a:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("Per-class ROC: Stack (proposed)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "roc_curve.png"), dpi=200)
    plt.close()
    print("saved roc_curve.png")


if __name__ == "__main__":
    fig_dca()
    fig_calibration()
    fig_cm()
    fig_roc()
