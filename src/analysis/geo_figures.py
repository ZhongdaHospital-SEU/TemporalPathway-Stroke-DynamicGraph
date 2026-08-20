# -*- coding: utf-8 -*-
"""GEO three-phase pathway atlas figures.

Inputs (from src/data/geo_pathway.py):
  data/processed/geo_pathway/activity_<GSE>.csv, pheno_<GSE>.csv, temporal_atlas.csv
Outputs:
  results/figures/pathway_atlas_heatmap.png   (acute 3h/5h/24h + subacute B/FU)
  results/figures/pathway_trajectories.png    (top acute & subacute pathways)
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

OUT = r"D:\TT paper\0811Temporal Pathway\data\processed\geo_pathway"
FIG = r"D:\TT paper\0811Temporal Pathway\results\figures"
os.makedirs(FIG, exist_ok=True)
sns.set_theme(style="whitegrid", font_scale=1.1)

K = 12  # top pathways per phase


def load():
    atlas = pd.read_csv(os.path.join(OUT, "temporal_atlas.csv"))
    act = {g: pd.read_csv(os.path.join(OUT, f"activity_{g}.csv"), index_col=0)
           for g in ["GSE37587", "GSE58294", "GSE16561", "GSE22255"]}
    ph = {g: pd.read_csv(os.path.join(OUT, f"pheno_{g}.csv"), index_col=0)
          for g in ["GSE37587", "GSE58294", "GSE16561", "GSE22255"]}
    return atlas, act, ph


def fig1(atlas, act, ph):
    """Heatmap: top significant pathways x time points (acute + subacute)."""
    acute = atlas[(atlas.dataset == "GSE58294") & (atlas.fdr < 0.05)].sort_values("fdr").head(K)
    sub = atlas[(atlas.dataset == "GSE37587") & (atlas.fdr < 0.05)].sort_values("fdr").head(K)
    rows = list(acute.pathway) + list(sub.pathway)
    if not rows:
        print("no significant pathways (FDR<0.05) for heatmap")
        return
    cols, col_tags = [], []
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
            sel = a.loc[pw, idx]
            mat[i, j] = sel.mean()
    mz = (mat - np.nanmean(mat, axis=1, keepdims=True)) / np.nanstd(mat, axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(11, 0.42 * len(rows) + 2.5))
    sns.heatmap(mz, cmap="RdBu_r", center=0, vmin=-2, vmax=2, ax=ax,
                xticklabels=[c[2] for c in cols],
                yticklabels=[p.split(":")[-1][:42] for p in rows], cbar_kws={"label": "z(ssGSEA)"})
    ax.set_title("Temporal pathway atlas: acute (GSE58294, cardioembolic) vs subacute (GSE37587)")
    if len(acute) and len(sub):
        ax.axvline(3, color="k", lw=1.2)
    ax.tick_params(axis="y", labelsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "pathway_atlas_heatmap.png"), dpi=200)
    plt.close()
    print("saved pathway_atlas_heatmap.png")


def fig2(atlas, act, ph):
    """Trajectory lines for top acute & subacute pathways."""
    acute = atlas[(atlas.dataset == "GSE58294") & (atlas.fdr < 0.05)].sort_values("fdr").head(6)
    sub = atlas[(atlas.dataset == "GSE37587") & (atlas.fdr < 0.05)].sort_values("fdr").head(6)
    n_panels = (1 if len(acute) else 0) + (1 if len(sub) else 0)
    if n_panels == 0:
        print("no significant pathways for trajectory figure")
        return
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
        axes[pi].set_title("Acute phase (GSE58294, 3h->24h)")
        axes[pi].set_ylabel("ssGSEA activity")
        axes[pi].legend(fontsize=7, loc="best")
        pi += 1
    if len(sub):
        a87 = act["GSE37587"]; p87 = ph["GSE37587"]
        for pw in sub.pathway:
            b = a87.loc[pw, p87[p87["time"] == "Baseline"].index]
            f = a87.loc[pw, p87[p87["time"] == "Follow-Up"].index]
            axes[pi].plot([0, 1], [b.mean(), f.mean()], marker="o",
                          label=pw.split(":")[-1][:40])
        axes[pi].set_xticks([0, 1]); axes[pi].set_xticklabels(["Baseline", "Follow-Up"])
        axes[pi].set_title("Subacute phase (GSE37587)")
        axes[pi].legend(fontsize=7, loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "pathway_trajectories.png"), dpi=200)
    plt.close()
    print("saved pathway_trajectories.png")


if __name__ == "__main__":
    atlas, act, ph = load()
    print(f"[figs] atlas rows: {len(atlas)}, sig: {(atlas.fdr < 0.05).sum()}")
    fig1(atlas, act, ph)
    fig2(atlas, act, ph)
