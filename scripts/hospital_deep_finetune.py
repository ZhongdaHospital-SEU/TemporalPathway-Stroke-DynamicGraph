# -*- coding: utf-8 -*-
"""Fine-tune the MIMIC-pretrained DG-GRU+TPR fold models on the 185 real cases (5-fold CV).
Compares: (i) pretrained + fine-tune, (ii) from-scratch on real cohort, (iii) pretrained frozen.
"""
from __future__ import annotations
import os, sys, time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import warnings, io, json
warnings.filterwarnings("ignore")

torch.set_num_threads(int(os.environ.get("FT_THREADS", "8")))
BASE = r"D:\TT paper\0811Temporal Pathway"
SRC = os.path.join(BASE, "src", "temporal_pathway")
sys.path.insert(0, SRC)
from train_gru import fold_stats, impute_seq, N_CLASSES
from train_dggru import DGGRUOrdinal, tpr_loss

WORK = os.path.join(BASE, "work")
EXT = os.path.join(BASE, "results", "external_validation")
DROP_M = ["hadm_id", "subject_id", "discharge_location", "outcome_ordinal", "hospital_expire_flag"]
N_EPOCHS = int(os.environ.get("FT_EPOCHS", "10"))
LR = 1e-4
BATCH = 64

def norm_cols(df):
    df = df.copy()
    df.columns = [c.replace("lab_", "") for c in df.columns]
    return df

SCHEMA = json.load(io.open(os.path.join(WORK, "harmonized_schema.json"), encoding="utf-8"))
FEAT_NAMES = [f"{l}_{st}" for l in [x["name"] for x in SCHEMA["labs"]] for st in SCHEMA["lab_stats"]] + \
             [f"{c}_{st}" for c in SCHEMA["channels"] for st in SCHEMA["channel_stats"]] + list(SCHEMA["demo_features"])

def load_mimic_feats():
    s = norm_cols(pd.read_csv(os.path.join(BASE, "data", "processed", "mimic_stroke", "mimic_harmonized_static.csv")))
    s = s[s["outcome_ordinal"].notna()].copy()
    feats = [c for c in s.columns if c not in DROP_M]
    X = s[feats].astype(np.float32)
    X = X.loc[:, X.notna().any()]
    return [c for c in X.columns], X.median()

def load_hospital():
    s = pd.read_csv(os.path.join(WORK, "hospital_static.csv"))
    s.columns = [c.replace("lab_", "") if c.startswith("lab_") else c for c in s.columns]
    feats = [c for c in s.columns if c in FEAT_NAMES]
    z = np.load(os.path.join(WORK, "hospital_seqs.npz"))
    seqs = {k[4:]: z[k] for k in z.files if k.startswith("seq_")}
    masks = {k[5:]: z[k] for k in z.files if k.startswith("mask_")}
    s = s[s["Patient ID"].isin(seqs.keys())].copy()
    ids = s["Patient ID"].astype(str).values
    Xs = np.stack([seqs[i] for i in ids]).astype(np.float32)
    M = np.stack([masks[i] for i in ids]).astype(np.float32)
    if M.ndim == 3:
        M = (M.sum(axis=-1) > 0).astype(np.float32)
    return ids, s, feats, Xs, M

def macro_auc_ovr(y_true, proba, classes):
    aucs = []
    for c in classes:
        if (y_true == c).sum() < 1:
            continue
        aucs.append(roc_auc_score((y_true == c).astype(int), proba[:, c]))
    return float(np.mean(aucs)) if aucs else float("nan")

def fit_fold(Xseq_tr, M_tr, S_tr, y_tr, Xseq_te, M_te, S_te, init_state=None,
             freeze=False, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    mean, std = fold_stats(Xseq_tr, M_tr, np.arange(len(y_tr)))
    Xtr = impute_seq(Xseq_tr, M_tr, mean, std)
    Xte = impute_seq(Xseq_te, M_te, mean, std)
    model = DGGRUOrdinal(n_seq=8, hidden=64, n_static=S_tr.shape[1], use_graph=True)
    if init_state is not None:
        model.load_state_dict(init_state)
    if freeze:
        for p in model.parameters():
            p.requires_grad = False
        # unfreeze head so labels can be learned
        for p in model.thresh.parameters():
            p.requires_grad = True
        for p in model.score.parameters():
            p.requires_grad = True
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=LR, weight_decay=1e-4)
    counts = np.bincount(y_tr, minlength=N_CLASSES).astype(np.float32)
    w = (len(y_tr) / (N_CLASSES * counts)).astype(np.float32)
    weight = torch.from_numpy(w)
    Xt = torch.from_numpy(Xtr); Mt = torch.from_numpy(M_tr)
    St = torch.from_numpy(S_tr.astype(np.float32)); Yt = torch.from_numpy(y_tr.astype(np.int64))
    n = len(y_tr)
    for epoch in range(N_EPOCHS):
        model.train(); perm = torch.randperm(n); tot, nb = 0.0, 0
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            logp, s, m = model(Xt[idx], Mt[idx], St[idx], return_traj=True)
            loss = F.nll_loss(logp, Yt[idx], weight=weight) + tpr_loss(s, m, 0.2, 0.05)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tot += float(loss.item()); nb += 1
    model.eval()
    with torch.no_grad():
        logp = model(torch.from_numpy(Xte), torch.from_numpy(M_te),
                     torch.from_numpy(S_te.astype(np.float32)))
    return np.nan_to_num(logp.exp().numpy(), nan=1.0 / N_CLASSES)

def main():
    mimic_feats, mimic_med = load_mimic_feats()
    ids, s, feats, Xs, M = load_hospital()
    shared = [c for c in mimic_feats if c in feats]
    X = s[shared].astype(np.float32)
    # fill all-NaN hospital lab columns with MIMIC medians (finite), others with column median
    X = X.apply(lambda c: c.fillna(c.median() if c.notna().any() else float(mimic_med[c.name])))
    y = np.minimum(s["mrs90"].values, 5).astype(int)
    poor = (s["mrs90"].values >= 3).astype(int)
    print(f"hospital n={len(ids)} shared={len(shared)} epochs={N_EPOCHS} lr={LR} threads={torch.get_num_threads()}", flush=True)
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    results = {"pretrain_finetune": [], "scratch": [], "pretrain_frozen": []}
    for f, (tr, te) in enumerate(skf.split(X, poor)):
        sc = StandardScaler().fit(X.iloc[tr])
        S_tr = sc.transform(X.iloc[tr]).astype(np.float32)
        S_te = sc.transform(X.iloc[te]).astype(np.float32)
        ck = os.path.join(EXT, f"hospital_fold_dggru_tpr_f{f}.pt")
        init = torch.load(ck, map_location="cpu")["model"] if os.path.exists(ck) else None
        for mode in results:
            if mode == "pretrain_finetune":
                p = fit_fold(Xs[tr], M[tr], S_tr, y[tr], Xs[te], M[te], S_te, init, False, 42 + f)
            elif mode == "scratch":
                p = fit_fold(Xs[tr], M[tr], S_tr, y[tr], Xs[te], M[te], S_te, None, False, 42 + f)
            else:
                p = fit_fold(Xs[tr], M[tr], S_tr, y[tr], Xs[te], M[te], S_te, init, True, 42 + f)
            bin_auc = roc_auc_score(poor[te], p[:, 3:].sum(1))
            ord_auc = macro_auc_ovr(y[te], p, list(range(6)))
            results[mode].append((bin_auc, ord_auc))
            print(f"  fold {f} {mode}: bin={bin_auc:.4f} ord={ord_auc:.4f}", flush=True)
    rows = []
    for mode, v in results.items():
        b = np.mean([x[0] for x in v]); o = np.mean([x[1] for x in v])
        rows.append({"mode": mode, "binary_auc": round(float(b), 3), "ordinal_macroauc": round(float(o), 3)})
        print(f"[{mode}] binary AUC {b:.3f} | ordinal macro-AUC {o:.3f}", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(EXT, "hospital_deep_finetune_results.csv"), index=False)
    print("DONE", flush=True)

if __name__ == "__main__":
    main()
