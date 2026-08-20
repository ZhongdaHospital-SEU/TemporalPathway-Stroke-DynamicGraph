# -*- coding: utf-8 -*-
"""DeLong paired test (canonical fastDeLong, Sun & Xu 2014) per class + mean."""
from __future__ import annotations
import os
import numpy as np
from scipy.stats import norm

RESULTS = r"D:\TT paper\0811Temporal Pathway\results"


def compute_midrank(x):
    """Midranks (tie-aware) for a 1-D array."""
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1)
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T + 1
    return T2


def fast_deLong(pred_sorted, m):
    """AUCs + covariance for k paired classifiers (columns of pred_sorted)."""
    n = pred_sorted.shape[0] - m
    pos = pred_sorted[:m]
    neg = pred_sorted[m:]
    k = pred_sorted.shape[1]
    tx = np.empty([k, m]); ty = np.empty([k, n]); tz = np.empty([k, m + n])
    for r in range(k):
        tx[r, :] = compute_midrank(pos[:, r])
        ty[r, :] = compute_midrank(neg[:, r])
        tz[r, :] = compute_midrank(pred_sorted[:, r])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    cov = sx / m + sy / n
    return aucs, cov


def delong_test(y_true, p1, p2):
    """Paired test between two binary score vectors. Returns dAUC, z, p."""
    y = np.asarray(y_true, dtype=bool)
    order = np.argsort(-y.astype(int), kind="stable")  # positives first
    m = int(y.sum())
    if m == 0 or m == len(y):
        return 0.0, 0.0, 1.0
    X = np.stack([p1[order], p2[order]], axis=1)
    aucs, cov = fast_deLong(X, m)
    d = aucs[1] - aucs[0]
    se = np.sqrt(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1])
    if se < 1e-12:
        return d, 0.0, 1.0
    z = d / se
    p = 2 * (1 - norm.cdf(abs(z)))
    return d, z, p


def main():
    st = np.load(os.path.join(RESULTS, "stack_oof.npz"))
    y = st["y"]
    xgb_ts = st["xgb_ts"]
    stack = st["stack2"]
    lines = ["=== DeLong paired test: stack2 vs XGB static+TS (per class) ==="]
    ds, zs, ps = [], [], []
    for cls in range(6):
        d, z, p = delong_test((y == cls).astype(int), xgb_ts[:, cls], stack[:, cls])
        ds.append(d); zs.append(z); ps.append(p)
        lines.append(f"class {cls}: dAUC={d:+.4f} z={z:.2f} p={p:.4f}")
    lines.append(f"mean dAUC={np.mean(ds):+.4f} mean z={np.mean(zs):.2f} "
                 f"min p={np.min(ps):.4f}")
    text = "\n".join(lines)
    print(text, flush=True)
    with open(os.path.join(RESULTS, "delong_result.txt"), "w", encoding="utf-8") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
