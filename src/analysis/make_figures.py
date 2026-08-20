# -*- coding: utf-8 -*-
"""Final publication figures: Figure 1A-1D, 2A-2D, 3A-3B (SVG + PNG).

Figure 1 (Methods):  1A study design + model architecture
                     1B temporal pathway atlas heatmap
                     1C pathway trajectories
                     1D recovery-score trajectories by outcome
Figure 2 (Results):  2A model comparison (macro-AUC)
                     2B confusion matrix (Stack)
                     2C per-class ROC (Stack)
                     2D DCA curves
Figure 3 (Clinical): 3A feature importance (XGB gain)
                     3B calibration curves
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
mpl.rcParams["svg.fonttype"] = "none"  # keep text as editable <text> in SVG
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import seaborn as sns
from sklearn.metrics import roc_curve, auc as roc_auc, confusion_matrix, ConfusionMatrixDisplay

BASE = r"D:\TT paper\0811Temporal Pathway"
RES = os.path.join(BASE, "results")
OUT = os.path.join(RES, "figures")
GEO = os.path.join(BASE, "data", "processed", "geo_pathway")
os.makedirs(OUT, exist_ok=True)
sns.set_theme(style="whitegrid", font_scale=1.1)
N_CLASSES = 6
LABELS = ["HOME", "HOME CARE", "REHAB", "SNF", "HOSPICE", "DIED"]


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".svg"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("saved " + name + ".svg/.png", flush=True)


def box(ax, x, y, w, h, text, fc, ec="black", fs=9, tc="black", lw=1.2):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.02",
                       fc=fc, ec=ec, lw=lw)
    ax.add_patch(b)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=tc, wrap=True)
    return b


def arrow(ax, x1, y1, x2, y2, color="0.25"):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14,
                        color=color, lw=1.4)
    ax.add_patch(a)


def fig1a():
    fig, ax = plt.subplots(figsize=(13.5, 7.5))
    ax.set_xlim(0, 13.5); ax.set_ylim(0, 7.5); ax.axis("off")
    ax.text(6.75, 7.15, "Temporal pathway-regularized dynamic graph learning for ordinal recovery prediction",
            ha="center", fontsize=12, fontweight="bold")
    # data sources
    box(ax, 0.2, 5.3, 3.4, 1.3, "MIMIC-IV v3.1 stroke ICU cohort\nn=2875, 64-step ICU sequences (14 features)\n+ 218 static features", "#DEEBF7")
    box(ax, 0.2, 2.6, 3.4, 1.9, "GEO stroke cohorts (4)\nGSE37587 / GSE58294\nGSE16561 / GSE22255", "#E2F0D9")
    box(ax, 0.2, 0.3, 3.4, 1.5, "Pathway gene sets\nKEGG / Reactome / GO-BP", "#FFF2CC")
    # pathway prior
    box(ax, 4.2, 3.0, 2.6, 1.5, "ssGSEA pathway activity\nTemporal pathway atlas\n(acute 3h-24h, subacute B-FU)", "#E2F0D9")
    # models
    box(ax, 7.3, 4.6, 2.7, 1.6, "XGB static / static+TS\n(5-fold OOF)", "#D9E1F2")
    box(ax, 7.3, 2.2, 2.7, 1.6, "DG-GRU + TPR\ndynamic feature graph\n+ pathway prior", "#FCE4D6")
    # stack
    box(ax, 10.6, 3.3, 2.5, 1.6, "Stacking\n(logistic, C=0.1)", "#D4380D", fs=10, tc="white")
    # outcome
    box(ax, 10.6, 0.4, 2.5, 2.2, "Ordinal outcome\nHOME(0) < HOME CARE(1)\n< REHAB(2) < SNF(3)\n< HOSPICE(4) < DIED(5)", "#FFF2CC")
    # arrows
    arrow(ax, 3.6, 5.9, 7.3, 5.4)
    arrow(ax, 3.6, 3.5, 4.2, 3.75)
    arrow(ax, 1.9, 2.6, 1.9, 1.8)
    arrow(ax, 3.0, 1.05, 4.6, 1.05); arrow(ax, 4.6, 1.05, 4.6, 3.0); arrow(ax, 4.6, 1.05, 7.3, 2.6)
    arrow(ax, 6.8, 3.6, 7.3, 3.6)
    arrow(ax, 10.0, 5.4, 10.6, 4.4)
    arrow(ax, 10.0, 3.0, 10.6, 4.1)
    arrow(ax, 11.85, 3.3, 11.85, 2.6)
    arrow(ax, 6.0, 2.6, 7.3, 2.2, color="0.6")
    ax.text(6.9, 2.35, "TPR", fontsize=9, color="0.3", ha="center")
    ax.text(6.75, 0.75, "Evaluation: acc / kappa / macro-AUC / NRI / IDI / DCA / calibration",
            ha="center", fontsize=9.5, style="italic")
    save(fig, "Figure1A")


def fig1b():
    atlas = pd.read_csv(os.path.join(GEO, "temporal_atlas.csv"))
    act = {g: pd.read_csv(os.path.join(GEO, f"activity_{g}.csv"), index_col=0)
           for g in ["GSE37587", "GSE58294"]}
    ph = {g: pd.read_csv(os.path.join(GEO, f"pheno_{g}.csv"), index_col=0)
          for g in ["GSE37587", "GSE58294"]}
    K = 12
    acute = atlas[(atlas.dataset == "GSE58294") & (atlas.fdr < 0.05)].sort_values("fdr").head(K)
    sub = atlas[(atlas.dataset == "GSE37587") & (atlas.fdr < 0.05)].sort_values("fdr").head(K)
    rows = list(acute.pathway) + list(sub.pathway)
    cols = []
    a94 = act["GSE58294"]; p94 = ph["GSE58294"]
    a87 = act["GSE37587"]; p87 = ph["GSE37587"]
    stroke94 = p94[p94["group"] == "Cardioembolic Stroke"]
    if len(acute):
        for t, tag in [("3", "3h"), ("5", "5h"), ("24", "24h")]:
            cols.append((a94, stroke94[stroke94["time"] == t].index, tag))
    if len(sub):
        for t, tag in [("Baseline", "Baseline"), ("Follow-Up", "Follow-Up")]:
            cols.append((a87, p87[p87["time"] == t].index, tag))
    mat = np.full((len(rows), len(cols)), np.nan)
    for j, (a, idx, _) in enumerate(cols):
        for i, pw in enumerate(rows):
            mat[i, j] = a.loc[pw, idx].mean()
    mz = (mat - np.nanmean(mat, axis=1, keepdims=True)) / np.nanstd(mat, axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(11, 0.42 * len(rows) + 2.5))
    sns.heatmap(mz, cmap="RdBu_r", center=0, vmin=-2, vmax=2, ax=ax,
                xticklabels=[c[2] for c in cols],
                yticklabels=[p.split(":")[-1][:42] for p in rows], cbar_kws={"label": "z(ssGSEA)"})
    ax.set_title("Temporal pathway atlas: acute (GSE58294) vs subacute (GSE37587)")
    if len(acute) and len(sub):
        ax.axvline(3, color="k", lw=1.2)
    ax.tick_params(axis="y", labelsize=8)
    save(fig, "Figure1B")


def fig1c():
    atlas = pd.read_csv(os.path.join(GEO, "temporal_atlas.csv"))
    act = {g: pd.read_csv(os.path.join(GEO, f"activity_{g}.csv"), index_col=0)
           for g in ["GSE37587", "GSE58294"]}
    ph = {g: pd.read_csv(os.path.join(GEO, f"pheno_{g}.csv"), index_col=0)
          for g in ["GSE37587", "GSE58294"]}
    acute = atlas[(atlas.dataset == "GSE58294") & (atlas.fdr < 0.05)].sort_values("fdr").head(6)
    sub = atlas[(atlas.dataset == "GSE37587") & (atlas.fdr < 0.05)].sort_values("fdr").head(6)
    n_panels = (1 if len(acute) else 0) + (1 if len(sub) else 0)
    if n_panels == 0:
        print("fig1c: no significant pathways"); return
    fig, axes = plt.subplots(1, n_panels, figsize=(6.5 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]
    pi = 0
    if len(acute):
        a94 = act["GSE58294"]; p94 = ph["GSE58294"]
        stroke94 = p94[p94["group"] == "Cardioembolic Stroke"]
        for pw in acute.pathway:
            means, sems = [], []
            for t in ["3", "5", "24"]:
                v = a94.loc[pw, stroke94[stroke94["time"] == t].index]
                means.append(v.mean()); sems.append(v.sem())
            axes[pi].errorbar(["3h", "5h", "24h"], means, yerr=sems, marker="o", capsize=3,
                              label=pw.split(":")[-1][:40])
        axes[pi].set_title("Acute phase (GSE58294, 3h-24h)")
        axes[pi].set_ylabel("ssGSEA activity")
        axes[pi].legend(fontsize=7, loc="best")
        pi += 1
    if len(sub):
        a87 = act["GSE37587"]; p87 = ph["GSE37587"]
        for pw in sub.pathway:
            b = a87.loc[pw, p87[p87["time"] == "Baseline"].index]
            f = a87.loc[pw, p87[p87["time"] == "Follow-Up"].index]
            axes[pi].plot([0, 1], [b.mean(), f.mean()], marker="o", label=pw.split(":")[-1][:40])
        axes[pi].set_xticks([0, 1]); axes[pi].set_xticklabels(["Baseline", "Follow-Up"])
        axes[pi].set_title("Subacute phase (GSE37587)")
        axes[pi].legend(fontsize=7, loc="best")
    save(fig, "Figure1C")


def fig1d():
    d = np.load(os.path.join(RES, "trajectory_scores.npz"))
    s, m, y = d["s"], d["mask"], d["y"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    cmap = plt.cm.viridis(np.linspace(0, 1, N_CLASSES))
    for k in range(N_CLASSES):
        idx = np.where(y == k)[0]
        lens = m[idx].sum(1)
        xs_all, ss_all = [], []
        for j in range(len(idx)):
            t = int(lens[j])
            xs_all.append(np.linspace(0, 1, t))
            ss_all.append(s[idx[j], :t])
        x = np.concatenate(xs_all); sv = np.concatenate(ss_all)
        bins = np.linspace(0, 1, 21)
        xb = np.clip(np.searchsorted(bins, x, "right") - 1, 0, 19)
        means = np.array([sv[xb == b].mean() if (xb == b).any() else np.nan for b in range(20)])
        axes[0].plot(bins[:-1] + 0.025, means, color=cmap[k], marker="o", ms=3,
                     label=f"{LABELS[k]} (n={len(idx)})")
    axes[0].set_title("Recovery score trajectory (DG-GRU+TPR)")
    axes[0].set_xlabel("Relative ICU time"); axes[0].set_ylabel("Recovery score s_t")
    axes[0].legend(fontsize=7, ncol=2)
    # calibration of score vs outcome ordinal mean
    ord_mean = (np.eye(N_CLASSES)[y] * np.arange(N_CLASSES)).sum(1)
    score_mean = (s * m).sum(1) / m.sum(1).clip(min=1)
    axes[1].scatter(ord_mean, score_mean, s=6, alpha=0.25, color="#4C72B0")
    axes[1].set_xlabel("Ordinal outcome (0-5)"); axes[1].set_ylabel("Mean recovery score")
    axes[1].set_title("Recovery score vs ordinal outcome")
    save(fig, "Figure1D")


def fig2a():
    import re
    rows = []
    base_txt = ""
    bpath = os.path.join(RES, "cv_baselines.txt")
    if os.path.exists(bpath):
        base_txt = open(bpath, encoding="utf-8").read()
    ORDER = [
        ("XGB static (full)", "0.8142", "#8c8c8c"),
        ("XGB static (ICU)", "0.8172", "#a0a0a0"),
        ("XGB static+TS", "0.8260", "#5b8ff9"),
        ("GRU seq-only", None, "#61c0a8"),
        ("GRU+static", None, "#61c0a8"),
        ("GRU+TPR", None, "#61c0a8"),
        ("DG-GRU", None, "#9254de"),
        ("DG-GRU+TPR", None, "#9254de"),
        ("Stack (proposed)", None, "#d4380d"),
    ]
    txt_cache = {}
    for label, hard, color in ORDER:
        if hard is not None:
            rows.append((label, float(hard), None, color)); continue
        if label == "Stack (proposed)":
            rows.append((label, 0.8304, None, color)); continue
        tag = {"GRU seq-only": "gru_seqonly", "GRU+static": "gru", "GRU+TPR": "gru_tpr",
               "DG-GRU": "dggru", "DG-GRU+TPR": "dggru_tpr"}[label]
        p = os.path.join(RES, tag + "_result.txt")
        if not os.path.exists(p):
            continue
        t = open(p, encoding="utf-8").read()
        m = re.search(r"auc: ([0-9.]+)(?:\+/-([0-9.]+))?", t)
        if m:
            rows.append((label, float(m.group(1)), float(m.group(2)) if m.group(2) else None, color))
    fig, ax = plt.subplots(figsize=(9, 6))
    labels = [r[0] for r in rows]; aucs = [r[1] for r in rows]
    sds = [r[2] if r[2] else 0.01 for r in rows]; colors = [r[3] for r in rows]
    y = np.arange(len(rows))
    ax.barh(y, aucs, xerr=sds, color=colors, capsize=3, height=0.6)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Macro AUC (5-fold OOF)")
    ax.set_xlim(0.75, 0.86)
    ax.axvline(0.8260, color="k", ls="--", lw=0.8)
    for i, a in enumerate(aucs):
        ax.text(a + 0.0015, i, f"{a:.2f}", va="center", fontsize=8)
    ax.set_title("Ordinal discharge-disposition prediction (6 classes)")
    save(fig, "Figure2A")


def fig2b():
    st = np.load(os.path.join(RES, "stack_oof.npz"))
    cm = confusion_matrix(st["y"], st["proba"].argmax(1))
    disp = ConfusionMatrixDisplay(cm, display_labels=LABELS)
    fig, ax = plt.subplots(figsize=(8.5, 7))
    disp.plot(ax=ax, cmap="Blues", colorbar=True)
    ax.set_title("Stack (proposed) confusion matrix (5-fold OOF)")
    save(fig, "Figure2B")


def fig2c():
    st = np.load(os.path.join(RES, "stack_oof.npz"))
    y = st["y"]
    fig, ax = plt.subplots(figsize=(7.5, 6))
    for k in range(N_CLASSES):
        yb = (y == k).astype(int)
        fpr, tpr, _ = roc_curve(yb, st["proba"][:, k])
        a = roc_auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=1.6, label=f"{LABELS[k]} (AUC={a:.2f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("Per-class ROC: Stack (proposed)")
    ax.legend(fontsize=8)
    save(fig, "Figure2C")


def fig2d():
    d = pd.read_csv(os.path.join(RES, "dca_curves.csv"))
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
    save(fig, "Figure2D")


def fig3a():
    top = pd.read_csv(os.path.join(RES, "shap_top20.csv"))
    g = pd.read_csv(os.path.join(RES, "feat_group_importance.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    t = top.iloc[::-1]
    axes[0].barh(range(len(t)), t["gain"], color="#4C72B0")
    axes[0].set_yticks(range(len(t)))
    axes[0].set_yticklabels(t["feature"], fontsize=8)
    axes[0].set_xlabel("XGB gain importance")
    axes[0].set_title("Top-20 features (XGB static+TS)")
    g2 = g.iloc[::-1]
    axes[1].barh(range(len(g2)), g2["gain"], color="#55A868")
    axes[1].set_yticks(range(len(g2)))
    axes[1].set_yticklabels(g2["group"], fontsize=9)
    axes[1].set_xlabel("Aggregated gain")
    axes[1].set_title("Feature-group importance")
    for i, v in enumerate(g2["gain"]):
        axes[1].text(v + 0.005, i, f"{v:.2f}", va="center", fontsize=8)
    save(fig, "Figure3A")


def fig3b():
    d = pd.read_csv(os.path.join(RES, "calibration.csv"))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True, sharey=True)
    for ax, k in zip(axes, [1, 2, 3]):
        for name, color in [("xgb_static+ts", "b"), ("Stack (proposed)", "r")]:
            sub = d[(d["cutoff"] == k) & (d["model"] == name)]
            ax.plot([0, 1], [0, 1], "k--", lw=1)
            ax.plot(sub["pred"], sub["obs"], "o-", color=color, ms=4, label=name)
        ax.set_title(f"P(y>={k})")
        ax.set_xlabel("Predicted"); ax.set_ylabel("Observed")
        ax.legend(fontsize=8)
    save(fig, "Figure3B")


if __name__ == "__main__":
    fig1a()
    fig1b()
    fig1c()
    fig1d()
    fig2a()
    fig2b()
    fig2c()
    fig2d()
    fig3a()
    fig3b()
