# -*- coding: utf-8 -*-
"""v2 sequence model: attention-pooled GRU + static MLP branch (+DG graph / TPR).

Fixes the two weak points of the plain GRU baseline:
  1. attention pooling over ALL valid ICU steps (not just the last hidden state)
  2. an MLP branch for the 218-306 static/lab features (not a single linear concat)
Optionally re-uses the dynamic feature graph and the temporal-pathway
regularization from train_dggru for a full 'DG-GRU + TPR' headline model.
"""
from __future__ import annotations
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, cohen_kappa_score
import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from train_gru import (load_data, to_tensor, fold_stats, impute_seq,
                       multiclass_auc, RESULTS, N_CLASSES, N_THRESH)
from train_dggru import DynamicFeatureGraph, tpr_loss

torch.manual_seed(42)
np.random.seed(42)


class SeqOrdinalV2(nn.Module):
    def __init__(self, n_seq: int, hidden: int = 64, n_static: int = 0,
                 dropout: float = 0.3, use_graph: bool = True,
                 bidirectional: bool = True):
        super().__init__()
        self.use_graph = use_graph
        if use_graph:
            self.dfg = DynamicFeatureGraph(n_seq)
        n_dir = 2 if bidirectional else 1
        self.gru = nn.GRU(n_seq, hidden, batch_first=True, num_layers=1,
                          bidirectional=bidirectional)
        self.ln = nn.LayerNorm(hidden * n_dir)
        self.proj = nn.Linear(hidden * n_dir, 128)
        self.act = nn.GELU()
        self.att = nn.Linear(128, 1)
        self.score = nn.Linear(128, 1)
        self.drop = nn.Dropout(dropout)
        if n_static > 0:
            self.static_mlp = nn.Sequential(
                nn.Linear(n_static, 128), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(128, 64), nn.GELU())
        self.head = nn.Linear(128 + (64 if n_static > 0 else 0), N_THRESH + 1)

    def forward(self, x, mask, static=None, return_traj=False):
        if self.use_graph:
            x, _ = self.dfg(x)
        lengths = mask.sum(1).clamp(min=1).long()
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True,
                                                   enforce_sorted=False)
        out, _ = self.gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True,
                                                  total_length=x.shape[1])  # (B,T,Hd)
        h = self.act(self.proj(self.ln(out)))                              # (B,T,128)
        # attention pooling over valid steps
        aw = self.att(h).squeeze(-1).masked_fill(mask == 0, -1e9)
        aw = torch.softmax(aw, dim=1)
        rep = (h * aw.unsqueeze(-1)).sum(1)                                # (B,128)
        if static is not None:
            rep = torch.cat([rep, self.static_mlp(static)], dim=1)
        out = self.head(rep)                                               # (B,6)
        thresh = out[:, :N_THRESH]
        s = out[:, N_THRESH:]
        cum = torch.cumsum(F.softplus(thresh), dim=1)
        logits = cum - s
        ls = F.logsigmoid(logits.double())
        lp0 = ls[:, 0:1]
        d = (ls[:, :-1] - ls[:, 1:]).clamp(max=-1e-8)
        lpk = ls[:, 1:] + torch.log(-torch.expm1(d))
        lplast = F.logsigmoid(-logits.double()[:, -1:])
        log_proba = torch.cat([lp0, lpk, lplast], dim=1).float()
        if return_traj:
            traj = self.score(h).squeeze(-1)                               # (B,T)
            return log_proba, traj, mask
        return log_proba


def train_fold(Xseq_tr_raw, M_tr, Xs_tr, y_tr, Xseq_te_raw, M_te, Xs_te,
               n_epochs=30, batch=128, lr=1e-3, seed=0, verbose=False,
               use_graph=True, bidirectional=True, lam_mono=0.0, lam_drift=0.0):
    mean, std = fold_stats(Xseq_tr_raw, M_tr, np.arange(len(y_tr)))
    Xseq_tr = impute_seq(Xseq_tr_raw, M_tr, mean, std)
    Xseq_te = impute_seq(Xseq_te_raw, M_te, mean, std)
    torch.manual_seed(seed)
    model = SeqOrdinalV2(n_seq=Xseq_tr.shape[2], hidden=64,
                         n_static=Xs_tr.shape[1], use_graph=use_graph,
                         bidirectional=bidirectional)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    counts = np.bincount(y_tr, minlength=N_CLASSES).astype(np.float32)
    w = (len(y_tr) / (N_CLASSES * counts)).astype(np.float32)
    weight = torch.from_numpy(w)
    n = len(y_tr)
    Xtr_t = torch.from_numpy(Xseq_tr); Mtr_t = torch.from_numpy(M_tr)
    S_tr = torch.from_numpy(Xs_tr.astype(np.float32))
    Y_tr = torch.from_numpy(y_tr.astype(np.int64))
    for epoch in range(n_epochs):
        model.train()
        perm = torch.randperm(n)
        tot, nb = 0.0, 0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            sb = S_tr[idx]
            log_proba, traj, m = model(Xtr_t[idx], Mtr_t[idx], sb, return_traj=True)
            loss = F.nll_loss(log_proba, Y_tr[idx], weight=weight)
            if lam_mono > 0 or lam_drift > 0:
                loss = loss + tpr_loss(traj, m, lam_mono, lam_drift)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss.item()); nb += 1
        if verbose and (epoch + 1) % 5 == 0:
            print(f"  epoch {epoch + 1}/{n_epochs} loss={tot / nb:.4f}")
    model.eval()
    with torch.no_grad():
        log_proba = model(torch.from_numpy(Xseq_te), torch.from_numpy(M_te),
                          torch.from_numpy(Xs_te.astype(np.float32)))
    return log_proba.exp().numpy()


def run(n_epochs=30, mode="v2_dggru_tpr", verbose=True):
    use_graph = "graph" in mode
    use_tpr = "tpr" in mode
    bidir = "bi" in mode
    lam1 = 0.2 if use_tpr else 0.0
    lam2 = 0.05 if use_tpr else 0.0
    seqs, masks, Xs, y, hadms = load_data(use_ts_static=("tsstatic" in mode))
    Xseq, M = to_tensor(seqs, masks, hadms)
    print(f"[{mode}] n={len(y)}, seq={Xseq.shape}, static={Xs.shape}", flush=True)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs, kaps, aucs = [], [], []
    oof = np.zeros((len(y), N_CLASSES))
    for f, (tr, te) in enumerate(skf.split(Xs, y)):
        sc = StandardScaler().fit(Xs.iloc[tr])
        Xtr_s = sc.transform(Xs.iloc[tr]); Xte_s = sc.transform(Xs.iloc[te])
        proba = train_fold(Xseq[tr], M[tr], Xtr_s, y[tr], Xseq[te], M[te], Xte_s,
                           n_epochs=n_epochs, seed=42 + f, verbose=verbose,
                           use_graph=use_graph, bidirectional=bidir,
                           lam_mono=lam1, lam_drift=lam2)
        oof[te] = proba
        pred = proba.argmax(1)
        accs.append(accuracy_score(y[te], pred))
        kaps.append(cohen_kappa_score(y[te], pred))
        aucs.append(multiclass_auc(y[te], proba))
        print(f"  fold {f}: acc={accs[-1]:.4f} kappa={kaps[-1]:.4f} auc={aucs[-1]:.4f}", flush=True)
    line = (f"{mode} | acc: {np.mean(accs):.4f}+/-{np.std(accs):.4f} | "
            f"kappa: {np.mean(kaps):.4f} | auc: {np.mean(aucs):.4f}+/-{np.std(aucs):.4f}")
    print(line, flush=True)
    with open(os.path.join(RESULTS, f"{mode}_result.txt"), "w", encoding="utf-8") as f:
        f.write(line + "\n")
    np.savez(os.path.join(RESULTS, f"{mode}_oof.npz"), y=y, proba=oof, hadm_id=hadms)


if __name__ == "__main__":
    n_ep = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    mode = sys.argv[2] if len(sys.argv) > 2 else "v2_dggru_tpr"
    run(n_epochs=n_ep, mode=mode)
