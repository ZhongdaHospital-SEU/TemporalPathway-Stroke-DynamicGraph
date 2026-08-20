# -*- coding: utf-8 -*-
"""Ordinal clinical evaluation: NRI / IDI / DCA / calibration.

Compares the proposed stacked model against the XGBoost static-only baseline
on the same 5-fold out-of-fold predictions (paired by patient).
"""
from __future__ import annotations
import os
import glob
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, roc_auc_score
from scipy import stats

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


def brier_multiclass(y, proba):
    yb = np.eye(N_CLASSES)[y]
    return float(np.mean((proba - yb) ** 2))


def somers_d(y, proba):
    """Ordinal discrimination: Somer's D of predicted mean ordinal score vs y."""
    exp = (proba * np.arange(N_CLASSES)).sum(1)
    return float(stats.kendalltau(y, exp).statistic)


def nri_binary(y, p_old, p_new, cutoff):
    """Category-free NRI for event y>=cutoff."""
    ev = y >= cutoff
    dp = p_new - p_old
    up_ev = np.mean(dp[ev] > 0); down_ev = np.mean(dp[ev] < 0)
    up_ne = np.mean(dp[~ev] > 0); down_ne = np.mean(dp[~ev] < 0)
    return (up_ev - down_ev) + (down_ne - up_ne)


def idi_binary(y, p_old, p_new, cutoff):
    ev = y >= cutoff
    return float((p_new[ev].mean() - p_old[ev].mean()) -
                 (p_new[~ev].mean() - p_old[~ev].mean()))


def dca_curve(y, p_risk, pt_grid):
    """Net benefit for treating when P(y>=cutoff)>pt (cutoff folded into p_risk)."""
    y = np.asarray(y).astype(bool)
    nb = []
    n = len(y)
    prev = y.mean()
    for pt in pt_grid:
        tp = ((p_risk > pt) & y).sum() / n
        fp = ((p_risk > pt) & ~y).sum() / n
        nb.append(tp - fp * pt / (1 - pt))
    return np.array(nb)


def calibration_bins(y, p_risk, n_bins=10):
    """Observed vs predicted cumulative risk by bin; return slope via logit."""
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.searchsorted(bins, p_risk, side="right") - 1, 0, n_bins - 1)
    out = []
    for b in range(n_bins):
        m = idx == b
        if m.sum() >= 5:
            out.append((float(p_risk[m].mean()), float(y[m].mean()), int(m.sum())))
    out = np.array(out)
    eps = 1e-6
    lr = stats.linregress(np.log(out[:, 0] / (1 - out[:, 0] + eps)),
                          np.log(out[:, 1] / (1 - out[:, 1] + eps)))
    return out, float(lr.slope), float(lr.rvalue)


def main():
    cv = np.load(os.path.join(RESULTS, "cv_predictions.npz"))
    y = cv["y_icu"]
    xgb_s = cv["xgb_icu_static"]
    xgb_ts = cv["xgb_icu_ts"]
    models = {"XGB static": xgb_s, "XGB static+TS": xgb_ts}
    # auto-discover sequence-model OOF predictions
    for fn in sorted(glob.glob(os.path.join(RESULTS, "*_oof.npz"))):
        tag = os.path.basename(fn)[:-8]
        if tag.startswith("xgb") or tag == "stack":
            continue
        d = np.load(fn)
        if "proba" not in d.files:
            print(f"  [skip] {tag}: no proba key", flush=True)
            continue
        if not np.array_equal(d["y"], y):
            print(f"  [skip] {tag}: y mismatch", flush=True)
            continue
        models[tag] = d["proba"]
    # stack_oof.npz holds stack1/stack2; use stack2 (best) as proposed model
    st_path = os.path.join(RESULTS, "stack_oof.npz")
    if os.path.exists(st_path):
        st = np.load(st_path)
        if np.array_equal(st["y"], y):
            models["Stack (proposed)"] = st["stack2"]

    p_old = models["XGB static+TS"]
    p_new = models["Stack (proposed)"] if "Stack (proposed)" in models else models["dggru_tpr"]
    name_new = "Stack (proposed)" if "Stack (proposed)" in models else "dggru_tpr"
    name_old = "XGB static+TS"

    lines = []
    lines.append("=== Ordinal discrimination (OOF, ICU subset n=%d) ===" % len(y))
    for name, p in models.items():
        pred = p.argmax(1)
        lines.append(f"{name:20s} acc={accuracy_score(y, pred):.4f} "
                     f"kappa={cohen_kappa_score(y, pred):.4f} "
                     f"macroAUC={multiclass_auc(y, p):.4f} "
                     f"SomersD={somers_d(y, p):.4f} "
                     f"Brier={brier_multiclass(y, p):.4f}")

    lines.append("")
    lines.append("=== Incremental value: ICU timeseries (XGB static+TS vs XGB static) ===")
    for k in range(1, N_CLASSES):
        nri = nri_binary(y, xgb_s[:, k:].sum(1), xgb_ts[:, k:].sum(1), k)
        idi = idi_binary(y, xgb_s[:, k:].sum(1), xgb_ts[:, k:].sum(1), k)
        lines.append(f"  cutoff y>={k}: category-free NRI={nri:+.4f}  IDI={idi:+.4f}")
    nris = [nri_binary(y, xgb_s[:, k:].sum(1), xgb_ts[:, k:].sum(1), k)
            for k in range(1, N_CLASSES)]
    lines.append(f"  mean NRI (cutoffs 1-5): {np.mean(nris):+.4f}")

    lines.append("")
    lines.append("=== Incremental value: pathway prior (DG-GRU+TPR vs DG-GRU) ===")
    if "dggru_tpr" in models and "dggru" in models:
        for k in range(1, N_CLASSES):
            nri = nri_binary(y, models["dggru"][:, k:].sum(1), models["dggru_tpr"][:, k:].sum(1), k)
            idi = idi_binary(y, models["dggru"][:, k:].sum(1), models["dggru_tpr"][:, k:].sum(1), k)
            lines.append(f"  cutoff y>={k}: category-free NRI={nri:+.4f}  IDI={idi:+.4f}")
        nris = [nri_binary(y, models["dggru"][:, k:].sum(1), models["dggru_tpr"][:, k:].sum(1), k)
                for k in range(1, N_CLASSES)]
        lines.append(f"  mean NRI (cutoffs 1-5): {np.mean(nris):+.4f}")

    # DCA curves (risk of unfavorable = P(y>=2): institutional care or worse)
    pt = np.linspace(0.01, 0.99, 99)
    rows = []
    for k in [2, 3, 5]:
        ev = (y >= k).astype(float)
        rows.append({"cutoff": k, "pt": pt,
                     "xgb_static": dca_curve(ev, xgb_ts[:, k:].sum(1), pt),
                     "proposed": dca_curve(ev, p_new[:, k:].sum(1), pt),
                     "treat_all": ev.mean() - (1 - ev.mean()) * pt / (1 - pt)})
    dca_rows = []
    for r in rows:
        for i, pti in enumerate(r["pt"]):
            dca_rows.append({"cutoff": r["cutoff"], "pt": pti,
                             "xgb_static": r["xgb_static"][i],
                             "proposed": r["proposed"][i],
                             "treat_all": r["treat_all"][i]})
    pd.DataFrame(dca_rows).to_csv(os.path.join(RESULTS, "dca_curves.csv"), index=False)
    # net benefit improvement at clinical thresholds 0.1/0.2/0.3
    for k in [2, 3, 5]:
        ev = (y >= k).astype(float)
        for pt0 in [0.1, 0.2, 0.3]:
            nb_x = dca_curve(ev, xgb_ts[:, k:].sum(1), np.array([pt0]))[0]
            nb_g = dca_curve(ev, p_new[:, k:].sum(1), np.array([pt0]))[0]
            lines.append(f"  DCA y>={k} pt={pt0}: NB XGB static+TS={nb_x:.4f} {name_new}={nb_g:.4f} "
                         f"(d={nb_g-nb_x:+.4f})")

    lines.append("")
    lines.append("=== Calibration (cumulative risk P(y>=k), deciles) ===")
    cal_rows = []
    for k in [1, 2, 3]:
        for name, p in [("xgb_static+ts", xgb_ts), (name_new, p_new)]:
            bins, slope, r = calibration_bins((y >= k).astype(float), p[:, k:].sum(1))
            lines.append(f"  y>={k} {name:11s} slope={slope:.3f} R={r:.3f}")
            for b in bins:
                cal_rows.append({"cutoff": k, "model": name,
                                 "pred": b[0], "obs": b[1], "n": int(b[2])})
    pd.DataFrame(cal_rows).to_csv(os.path.join(RESULTS, "calibration.csv"), index=False)

    text = "\n".join(lines)
    print(text, flush=True)
    with open(os.path.join(RESULTS, "clinical_eval.txt"), "w", encoding="utf-8") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
