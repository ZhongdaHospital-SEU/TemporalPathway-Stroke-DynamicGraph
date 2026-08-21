# -*- coding: utf-8 -*-
"""Deep-model external validation dry-run: MIMIC harmonized -> 185-case hospital cohort.
Re-trains the study deep models (DG-GRU+TPR, GRU+TPR) on MIMIC harmonized 64x8 sequences
+ shared static features, then transfers directly to the hospital-cohort tensors built
from the ENG template (work/hospital_build.py).
"""
from __future__ import annotations
import os, sys, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, cohen_kappa_score, roc_auc_score
import warnings, io, json
warnings.filterwarnings("ignore")

torch.set_num_threads(int(os.environ.get("HOSP_THREADS", "6")))

BASE = r"D:\TT paper\0811Temporal Pathway"
SRC = os.path.join(BASE, "src", "temporal_pathway")
sys.path.insert(0, SRC)
from train_gru import (fold_stats, impute_seq, multiclass_auc, N_CLASSES, N_THRESH, SeqOrdinal)
from train_dggru import (DynamicFeatureGraph, DGGRUOrdinal, tpr_loss)

WORK = os.path.join(BASE, "work")
SCHEMA = json.load(io.open(os.path.join(WORK, "harmonized_schema.json"), encoding="utf-8"))
FEAT_NAMES = [f"{l}_{st}" for l in [x["name"] for x in SCHEMA["labs"]] for st in SCHEMA["lab_stats"]] + [f"{c}_{st}" for c in SCHEMA["channels"] for st in SCHEMA["channel_stats"]] + list(SCHEMA["demo_features"])
EXT = os.path.join(BASE, "results", "external_validation")
os.makedirs(EXT, exist_ok=True)
N_EPOCHS = int(os.environ.get("HOSP_EPOCHS", "30"))
BATCH = 64
LR = 1e-3

torch.manual_seed(42)
np.random.seed(42)
DROP_M = ["hadm_id", "subject_id", "discharge_location", "outcome_ordinal", "hospital_expire_flag"]


def norm_cols(df):
    df = df.copy()
    df.columns = [c.replace("lab_", "") for c in df.columns]
    return df


def load_mimic():
    p = os.path.join(BASE, "data", "processed", "mimic_stroke")
    z = np.load(os.path.join(p, "mimic_harmonized_seqs.npz"))
    seqs = {int(k[4:]): z[k] for k in z.files if k.startswith("seq_")}
    masks = {int(k[5:]): z[k] for k in z.files if k.startswith("mask_")}
    s = norm_cols(pd.read_csv(os.path.join(p, "mimic_harmonized_static.csv")))
    s = s[s["outcome_ordinal"].notna()].copy()
    s = s[s["hadm_id"].isin(seqs.keys())].copy()
    ids = s["hadm_id"].values.astype(np.int64)
    y = s["outcome_ordinal"].astype(int).values
    feats = [c for c in s.columns if c not in DROP_M]
    assert len(feats) == 173, len(feats)
    X = s[feats].astype(np.float32)
    allnan = X.isna().all()
    keep = [c for c in feats if not allnan[c]]
    X = X[keep]
    feats = keep
    Xs = np.stack([seqs[int(h)] for h in ids]).astype(np.float32)
    M = np.stack([masks[int(h)] for h in ids]).astype(np.float32)
    if M.ndim == 3:
        M = (M.sum(axis=-1) > 0).astype(np.float32)
    return ids, y, feats, X, Xs, M


def load_hospital():
    s = pd.read_csv(os.path.join(WORK, "hospital_static.csv"))
    s.columns = [c.replace("lab_", "") if c.startswith("lab_") else c for c in s.columns]
    feats = [c for c in s.columns if c in FEAT_NAMES]
    ids = s["Patient ID"].astype(str).values
    z = np.load(os.path.join(WORK, "hospital_seqs.npz"))
    seqs = {k[4:]: z[k] for k in z.files if k.startswith("seq_")}
    masks = {k[5:]: z[k] for k in z.files if k.startswith("mask_")}
    s = s[s["Patient ID"].isin(seqs.keys())].copy()
    ids = s["Patient ID"].astype(str).values
    y = s["mrs90"].values.astype(np.float32)
    X = s[feats].astype(np.float32)
    Xs = np.stack([seqs[i] for i in ids]).astype(np.float32)
    M = np.stack([masks[i] for i in ids]).astype(np.float32)
    if M.ndim == 3:
        M = (M.sum(axis=-1) > 0).astype(np.float32)
    extra = s[["mrs90", "death", "lost", "ivt", "mt", "nihss", "age", "gcs_first"]].copy()
    return ids, y, feats, X, Xs, M, extra


def agg64(proba6):
    p4 = np.zeros((proba6.shape[0], 4), dtype=np.float32)
    p4[:, 0] = proba6[:, 0]
    p4[:, 1] = proba6[:, 1] + proba6[:, 2]
    p4[:, 2] = proba6[:, 3]
    p4[:, 3] = proba6[:, 4] + proba6[:, 5]
    return p4


def macro_auc_ovr(y_true, proba, classes):
    aucs = []
    for c in classes:
        if (y_true == c).sum() < 1:
            continue
        aucs.append(roc_auc_score((y_true == c).astype(int), proba[:, c]))
    return float(np.mean(aucs)) if aucs else float("nan")


def train_fold_dggru(Xseq_tr_raw, M_tr, Xs_tr, y_tr, Xseq_te_raw, M_te, Xs_te, seed=0,
                     use_graph=True, lam_mono=0.2, lam_drift=0.05):
    mean, std = fold_stats(Xseq_tr_raw, M_tr, np.arange(len(y_tr)))
    Xseq_tr = impute_seq(Xseq_tr_raw, M_tr, mean, std)
    Xseq_te = impute_seq(Xseq_te_raw, M_te, mean, std)
    torch.manual_seed(seed); np.random.seed(seed)
    model = DGGRUOrdinal(n_seq=Xseq_tr.shape[2], hidden=64, n_static=Xs_tr.shape[1],
                         use_graph=use_graph)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    counts = np.bincount(y_tr, minlength=N_CLASSES).astype(np.float32)
    w = (len(y_tr) / (N_CLASSES * counts)).astype(np.float32)
    weight = torch.from_numpy(w)
    Xtr_t = torch.from_numpy(Xseq_tr); Mtr_t = torch.from_numpy(M_tr)
    S_tr = torch.from_numpy(Xs_tr.astype(np.float32)); Y_tr = torch.from_numpy(y_tr.astype(np.int64))
    n = len(y_tr)
    for epoch in range(N_EPOCHS):
        model.train(); perm = torch.randperm(n); tot, nb = 0.0, 0
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            log_proba = model(Xtr_t[idx], Mtr_t[idx], S_tr[idx])
            loss = F.nll_loss(log_proba, Y_tr[idx], weight=weight)
            if use_graph or lam_mono > 0:
                _, s, m = model(Xtr_t[idx], Mtr_t[idx], S_tr[idx], return_traj=True)
                loss = loss + tpr_loss(s, m, lam_mono, lam_drift)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tot += float(loss.item()); nb += 1
        if (epoch + 1) % 10 == 0:
            print(f"    epoch {epoch+1}/{N_EPOCHS} loss={tot/nb:.4f}", flush=True)
    model.eval()
    with torch.no_grad():
        log_proba = model(torch.from_numpy(Xseq_te), torch.from_numpy(M_te),
                          torch.from_numpy(Xs_te.astype(np.float32)))
    return np.nan_to_num(log_proba.exp().numpy(), nan=1.0 / N_CLASSES), model, mean, std


def train_fold_gru(Xseq_tr_raw, M_tr, Xs_tr, y_tr, Xseq_te_raw, M_te, Xs_te, seed=0):
    mean, std = fold_stats(Xseq_tr_raw, M_tr, np.arange(len(y_tr)))
    Xseq_tr = impute_seq(Xseq_tr_raw, M_tr, mean, std)
    Xseq_te = impute_seq(Xseq_te_raw, M_te, mean, std)
    torch.manual_seed(seed); np.random.seed(seed)
    model = SeqOrdinal(n_seq=Xseq_tr.shape[2], hidden=64, n_static=Xs_tr.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    counts = np.bincount(y_tr, minlength=N_CLASSES).astype(np.float32)
    w = (len(y_tr) / (N_CLASSES * counts)).astype(np.float32)
    weight = torch.from_numpy(w)
    Xtr_t = torch.from_numpy(Xseq_tr); Mtr_t = torch.from_numpy(M_tr)
    S_tr = torch.from_numpy(Xs_tr.astype(np.float32)); Y_tr = torch.from_numpy(y_tr.astype(np.int64))
    n = len(y_tr)
    for epoch in range(N_EPOCHS):
        model.train(); perm = torch.randperm(n); tot, nb = 0.0, 0
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            log_proba = model(Xtr_t[idx], Mtr_t[idx], S_tr[idx])
            loss = F.nll_loss(log_proba, Y_tr[idx], weight=weight)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tot += float(loss.item()); nb += 1
        if (epoch + 1) % 10 == 0:
            print(f"    epoch {epoch+1}/{N_EPOCHS} loss={tot/nb:.4f}", flush=True)
    model.eval()
    with torch.no_grad():
        log_proba = model(torch.from_numpy(Xseq_te), torch.from_numpy(M_te),
                          torch.from_numpy(Xs_te.astype(np.float32)))
    return np.nan_to_num(log_proba.exp().numpy(), nan=1.0 / N_CLASSES), model, mean, std


def predict_transfer(model, mean, std, Xseq, M, Xs, batch=512):
    Xseq_imp = impute_seq(Xseq, M, mean, std)
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(Xseq), batch):
            xb = torch.from_numpy(Xseq_imp[i:i + batch])
            mb = torch.from_numpy(M[i:i + batch])
            sb = torch.from_numpy(Xs[i:i + batch].astype(np.float32))
            out.append(np.nan_to_num(model(xb, mb, sb).exp().numpy(), nan=1.0 / N_CLASSES))
    return np.concatenate(out, axis=0)


def main():
    models = sys.argv[1:] if len(sys.argv) > 1 else ["dggru_tpr", "gru_tpr"]
    print(f"=== Hospital-cohort deep transfer dry-run === models={models} epochs={N_EPOCHS} threads={torch.get_num_threads()}", flush=True)
    t0 = time.time()
    mimic = load_mimic()
    print(f"  MIMIC loaded n={len(mimic[1])} feats={len(mimic[2])} [{time.time()-t0:.0f}s]", flush=True)
    hosp = load_hospital()
    ids_h, y_h, feats_h, X_h, Xs_h, M_h, extra = hosp
    print(f"  hospital loaded n={len(ids_h)} feats={len(feats_h)} seq={Xs_h.shape} [{time.time()-t0:.0f}s]", flush=True)

    shared = [c for c in mimic[2] if c in set(feats_h)]
    print("  shared features:", len(shared), flush=True)
    Xm = mimic[3][shared].fillna(mimic[3][shared].median())
    Xh = X_h[shared].astype(np.float32).fillna(mimic[3][shared].median().to_dict())
    mimic = (mimic[0], mimic[1], shared, Xm, mimic[4], mimic[5])
    hosp = (ids_h, y_h, shared, Xh, Xs_h, M_h, extra)

    poor_h = (y_h >= 3).astype(int)
    y6 = np.minimum(y_h, 5).astype(int)
    ok = (~np.isnan(y_h)) & (M_h.sum(axis=1) >= 4)
    print(f"  hospital ok={int(ok.sum())} poor rate={poor_h.mean():.3f}", flush=True)
    print("  mRS dist:", pd.Series(y_h).value_counts().sort_index().to_dict(), flush=True)

    rows = []
    for mname in models:
        out_npz = os.path.join(EXT, f"hospital_deep_transfer_{mname}.npz")
        if os.path.exists(out_npz):
            print(f"===== {mname}: output exists, skip =====", flush=True)
            continue
        print(f"\n===== {mname} =====", flush=True)
        use_graph = mname.startswith("dggru")
        use_tpr = mname.endswith("_tpr")
        lam1 = 0.2 if use_tpr else 0.0
        lam2 = 0.05 if use_tpr else 0.0
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        oof = np.zeros((len(mimic[1]), N_CLASSES), dtype=np.float32)
        fold_models = []
        ts = time.time()
        for f, (tr, te) in enumerate(skf.split(Xm, mimic[1])):
            sc = StandardScaler().fit(Xm.iloc[tr])
            Xtr_s = sc.transform(Xm.iloc[tr]).astype(np.float32)
            Xte_s = sc.transform(Xm.iloc[te]).astype(np.float32)
            ck = os.path.join(EXT, f"hospital_fold_{mname}_f{f}.pt")
            if os.path.exists(ck):
                print(f"  fold {f}: resume from checkpoint", flush=True)
                ckpt = torch.load(ck, map_location="cpu")
                model = ckpt["model_obj"]
                proba = ckpt["oof_proba"]
                sc = ckpt["scaler"]
                mean, std = ckpt["mean"], ckpt["std"]
            else:
                if use_graph or use_tpr:
                    proba, model, mean, std = train_fold_dggru(
                        mimic[4][tr], mimic[5][tr], Xtr_s, mimic[1][tr],
                        mimic[4][te], mimic[5][te], Xte_s, seed=42 + f,
                        use_graph=use_graph, lam_mono=lam1, lam_drift=lam2)
                else:
                    proba, model, mean, std = train_fold_gru(
                        mimic[4][tr], mimic[5][tr], Xtr_s, mimic[1][tr],
                        mimic[4][te], mimic[5][te], Xte_s, seed=42 + f)
            oof[te] = proba
            fold_models.append({"model": model, "scaler": sc, "mean": mean, "std": std})
            if not os.path.exists(ck):
                torch.save({"model": model.state_dict(), "scaler": sc, "mean": mean, "std": std,
                            "oof_proba": proba, "model_obj": model}, ck)
            print(f"  fold {f}: auc={multiclass_auc(mimic[1][te], proba):.4f} [{time.time()-ts:.0f}s]", flush=True)

        m_auc = multiclass_auc(mimic[1], oof)
        print(f"  MIMIC OOF {mname}: macro_auc={m_auc:.4f}", flush=True)

        proba6 = np.zeros((len(ids_h), N_CLASSES), dtype=np.float32)
        for fm in fold_models:
            Xs_std = fm["scaler"].transform(Xh).astype(np.float32)
            proba6 += predict_transfer(fm["model"], fm["mean"], fm["std"], Xs_h, M_h, Xs_std)
        proba6 /= float(len(fold_models))
        proba6 = np.nan_to_num(proba6, nan=1.0 / N_CLASSES)

        p_poor = proba6[:, 3:].sum(1)
        auc_bin = roc_auc_score(poor_h[ok], p_poor[ok]) if len(np.unique(poor_h[ok])) > 1 else float("nan")
        auc_ord = macro_auc_ovr(y6[ok], proba6[ok], list(range(6)))
        acc = accuracy_score(y6[ok], proba6[ok].argmax(1))
        kap = cohen_kappa_score(y6[ok], proba6[ok].argmax(1))
        # sensitivity: exclude 3 lost-FU
        notlost = (extra["lost"].values.astype(int) == 0) & ok
        auc_bin_nl = roc_auc_score(poor_h[notlost], p_poor[notlost]) if len(np.unique(poor_h[notlost])) > 1 else float("nan")
        print(f"  HOSP transfer {mname}: n_ok={int(ok.sum())} poor_auc={auc_bin:.4f} "
              f"ordinal_macro={auc_ord:.4f} acc={acc:.4f} kappa={kap:.4f} "
              f"poor_auc_noLostFU={auc_bin_nl:.4f}", flush=True)

        np.savez(out_npz, proba6=proba6, y6=y6, poor_h=poor_h, mrs90=y_h, ok=ok,
                 ids=ids_h, p_poor=p_poor)
        rows.append(dict(model=mname, mimic_oof_auc=float(m_auc), n_hosp=int(ok.sum()),
                         hosp_binary_auc=float(auc_bin), hosp_ordinal_macroauc=float(auc_ord),
                         hosp_acc=float(acc), hosp_kappa=float(kap),
                         hosp_binary_auc_excl_lost=float(auc_bin_nl)))
        pd.DataFrame(rows).to_csv(os.path.join(EXT, "hospital_deep_transfer_results.csv"), index=False)
        print(f"  [{mname} total {time.time()-ts:.0f}s]", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
