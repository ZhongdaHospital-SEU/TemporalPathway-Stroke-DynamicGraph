# -*- coding: utf-8 -*-
"""Recovery-score trajectories per discharge-outcome class (Fig 4).

Trains DG-GRU+TPR on the full ICU cohort (fixed seed), extracts the
per-step recovery score s_t for every patient, and plots mean trajectories
(relative time axis) stratified by ordinal outcome, alongside observed GCS.
"""
from __future__ import annotations
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, r"D:\TT paper\0811Temporal Pathway\src\temporal_pathway")
from train_gru import load_data, to_tensor, fold_stats, impute_seq
from train_dggru import DGGRUOrdinal
from train_dggru import tpr_loss

BASE = r"D:\TT paper\0811Temporal Pathway"
RESULTS = os.path.join(BASE, "results")
FIG = os.path.join(RESULTS, "figures")
os.makedirs(FIG, exist_ok=True)
N_CLASSES = 6
LABELS = ["HOME", "HOME CARE", "REHAB", "SNF", "HOSPICE", "DIED"]


def train_model(Xseq, M, Xs, y, n_epochs=20):
    mean, std = fold_stats(Xseq, M, np.arange(len(y)))
    Xs2 = impute_seq(Xseq, M, mean, std)
    torch.manual_seed(42)
    model = DGGRUOrdinal(n_seq=Xs2.shape[2], hidden=64, n_static=Xs.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    counts = np.bincount(y, minlength=N_CLASSES).astype(np.float32)
    w = (len(y) / (N_CLASSES * counts)).astype(np.float32)
    weight = torch.from_numpy(w)
    Xt = torch.from_numpy(Xs2); Mt = torch.from_numpy(M)
    St = torch.from_numpy(np.asarray(Xs, dtype=np.float32)); Yt = torch.from_numpy(y.astype(np.int64))
    n = len(y)
    for epoch in range(n_epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, 256):
            idx = perm[i:i + 256]
            logp, s, m = model(Xt[idx], Mt[idx], St[idx], return_traj=True)
            loss = F.nll_loss(logp, Yt[idx], weight=weight) + tpr_loss(s, m, 0.2, 0.05)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    model.eval()
    with torch.no_grad():
        _, s_all, m_all = model(Xt, Mt, St, return_traj=True)
    return s_all.numpy(), m_all.numpy()


def main():
    seqs, masks, Xs, y, hadms = load_data()
    Xseq, M = to_tensor(seqs, masks, hadms)
    print(f"[traj] n={len(y)}", flush=True)
    s, m = train_model(Xseq, M, Xs, y)
    # observed GCS total per patient (from raw sequences)
    gcs = Xseq[:, :, 1] + Xseq[:, :, 0] + Xseq[:, :, 2]  # motor+eye+verbal rough total
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    cmap = plt.cm.viridis(np.linspace(0, 1, N_CLASSES))
    for k in range(N_CLASSES):
        idx = np.where(y == k)[0]
        # relative time: x = step / length
        lens = M[idx].sum(1)
        xs_all, ss_all, gs_all = [], [], []
        for j in range(len(idx)):
            t = int(lens[j])
            xs_all.append(np.linspace(0, 1, t))
            ss_all.append(s[idx[j], :t])
            gs_all.append(gcs[idx[j], :t])
        x = np.concatenate(xs_all); sv = np.concatenate(ss_all); gv = np.concatenate(gs_all)
        # bin into 20 quantiles of x
        bins = np.linspace(0, 1, 21)
        xb = np.clip(np.searchsorted(bins, x, "right") - 1, 0, 19)
        means = np.array([sv[xb == b].mean() if (xb == b).any() else np.nan for b in range(20)])
        gmeans = np.array([gv[xb == b].mean() if (xb == b).any() else np.nan for b in range(20)])
        axes[0].plot(bins[:-1] + 0.025, means, color=cmap[k], marker="o", ms=3,
                     label=f"{LABELS[k]} (n={len(idx)})")
        axes[1].plot(bins[:-1] + 0.025, gmeans, color=cmap[k], marker="o", ms=3,
                     label=LABELS[k])
    axes[0].set_title("Recovery score trajectory (DG-GRU+TPR)")
    axes[0].set_xlabel("Relative ICU time"); axes[0].set_ylabel("Recovery score s_t")
    axes[1].set_title("Observed GCS trajectory")
    axes[1].set_xlabel("Relative ICU time"); axes[1].set_ylabel("GCS total (eye+motor+verbal)")
    for ax in axes:
        ax.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "recovery_trajectories.png"), dpi=200)
    plt.close()
    print("saved recovery_trajectories.png", flush=True)
    # also save the score matrix for further use
    np.savez(os.path.join(RESULTS, "trajectory_scores.npz"), s=s, mask=m, y=y)


if __name__ == "__main__":
    main()
