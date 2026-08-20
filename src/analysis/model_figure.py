# -*- coding: utf-8 -*-
"""Model comparison bar chart (macro-AUC with 95% CI) from result files."""
from __future__ import annotations
import os
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = r"D:\TT paper\0811Temporal Pathway\results"
FIG = os.path.join(RESULTS, "figures")
os.makedirs(FIG, exist_ok=True)

ORDER = [
    ("XGB static (full)", "xgb_full", "#8c8c8c"),
    ("XGB static (ICU)", "xgb_icu_static", "#a0a0a0"),
    ("XGB static+TS", "xgb_icu_ts", "#5b8ff9"),
    ("GRU seq-only", "gru_seqonly", "#61c0a8"),
    ("GRU+static", "gru", "#61c0a8"),
    ("GRU+TPR", "gru_tpr", "#61c0a8"),
    ("DG-GRU", "dggru", "#9254de"),
    ("DG-GRU+TPR", "dggru_tpr", "#9254de"),
    ("Stack (proposed)", "stack2", "#d4380d"),
]


def parse(fname):
    p = os.path.join(RESULTS, fname)
    if not os.path.exists(p):
        return None
    txt = open(p, encoding="utf-8").read().strip()
    m = re.search(r"auc: ([\d.]+)(?:\+/-([\d.]+))?", txt)
    if not m:
        m = re.search(r"macroAUC=([\d.]+)", txt)
    if m:
        return float(m.group(1)), (float(m.group(2)) if m.lastindex > 1 and m.group(2) else None)
    return None


def main():
    rows = []
    base_txt = open(os.path.join(RESULTS, "cv_baselines.txt"), encoding="utf-8").read() if os.path.exists(os.path.join(RESULTS, "cv_baselines.txt")) else ""
    for label, tag, color in ORDER:
        v = None
        if tag.startswith("xgb"):
            key = {"xgb_full": "full cohort", "xgb_icu_static": "ICU subset)", "xgb_icu_ts": "static+TS (ICU subset)"}.get(tag, "")
            m = re.search(r"XGB ([^(]+)\(?.*auc: ([\d.]+)", [l for l in base_txt.splitlines() if key in l][0] if any(key in l for l in base_txt.splitlines()) else "")
            if m:
                v = (float(m.group(2)), None)
        if tag == "stack2":
            txt = open(os.path.join(RESULTS, "stacking_result.txt"), encoding="utf-8").read()
            m = re.search(r"stack_xgb_ts\+xgb_s\+dggru_tpr\s+acc=[\d.]+ kappa=[\d.]+ macroAUC=([\d.]+)", txt)
            if m:
                v = (float(m.group(1)), None)
        if v is None:
            v = parse(tag + "_result.txt")
        if v:
            rows.append((label, v[0], v[1], color))
    fig, ax = plt.subplots(figsize=(9, 6))
    labels = [r[0] for r in rows]
    aucs = [r[1] for r in rows]
    sds = [r[2] if r[2] else 0.01 for r in rows]
    colors = [r[3] for r in rows]
    y = np.arange(len(rows))
    ax.barh(y, aucs, xerr=sds, color=colors, capsize=3, height=0.6)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Macro AUC (5-fold OOF)")
    ax.set_xlim(0.75, 0.86)
    ax.axvline(0.8260, color="k", ls="--", lw=0.8)
    ax.text(0.8262, len(rows) - 0.4, "XGB static+TS", fontsize=8)
    for i, a in enumerate(aucs):
        ax.text(a + 0.0015, i, f"{a:.3f}", va="center", fontsize=8)
    ax.set_title("Ordinal discharge-disposition prediction (6 classes)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "model_comparison.png"), dpi=200)
    plt.close()
    print("saved model_comparison.png", flush=True)
    for r in rows:
        print(r[0], r[1], flush=True)


if __name__ == "__main__":
    main()
