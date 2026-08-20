# -*- coding: utf-8 -*-
"""Aggregate all model result files into a comparison table (markdown + csv)."""
from __future__ import annotations
import os
import re
import numpy as np
import pandas as pd

RESULTS = r"D:\TT paper\0811Temporal Pathway\results"


def parse_result(fname):
    with open(os.path.join(RESULTS, fname), "r", encoding="utf-8") as f:
        line = f.read().strip().replace(" ordinal |", " |")
    m = re.match(r"([\w+-]+) \| acc: ([\d.+-]+)\+/-([\d.]+) \| kappa: ([\d.]+) \| auc: ([\d.+-]+)\+/-([\d.]+)", line)
    if not m:
        return None
    return {"model": m.group(1), "acc": float(m.group(2)), "acc_sd": float(m.group(3)),
            "kappa": float(m.group(4)), "auc": float(m.group(5)), "auc_sd": float(m.group(6))}


def main():
    rows = []
    for fn in sorted(os.listdir(RESULTS)):
        if fn.endswith("_result.txt"):
            r = parse_result(fn)
            if r:
                rows.append(r)
    # also parse baseline txt
    base = os.path.join(RESULTS, "cv_baselines.txt")
    if os.path.exists(base):
        with open(base, "r", encoding="utf-8") as f:
            for line in f.read().strip().splitlines():
                m = re.match(r"XGB ([^(]+)\(?.*\| acc: ([\d.]+) \| kappa: ([\d.]+) \| auc: ([\d.]+)", line)
                if m:
                    rows.append({"model": "XGB " + m.group(1).strip(), "acc": float(m.group(2)),
                                 "acc_sd": np.nan, "kappa": float(m.group(3)),
                                 "auc": float(m.group(4)), "auc_sd": np.nan})
    # stack (proposed) from stacking_result.txt
    st_path = os.path.join(RESULTS, "stacking_result.txt")
    if os.path.exists(st_path):
        with open(st_path, "r", encoding="utf-8") as f:
            for line in f.read().strip().splitlines():
                m = re.search(r"stack_xgb_ts\+xgb_s\+dggru_tpr\s+acc=([\d.]+) kappa=([\d.]+) macroAUC=([\d.]+)", line)
                if m:
                    rows.append({"model": "Stack (proposed)", "acc": float(m.group(1)),
                                 "acc_sd": np.nan, "kappa": float(m.group(2)),
                                 "auc": float(m.group(3)), "auc_sd": np.nan})
    df = pd.DataFrame(rows)
    if df.empty:
        print("no results yet")
        return
    order = ["xgb_full", "xgb_icu_static", "xgb_icu_ts", "gru_seqonly", "gru",
             "gru_tsstatic", "dggru", "dggru_tpr", "v2_dggru_tpr", "v2_tsstatic_bi"]
    df["order"] = df["model"].apply(lambda m: next((i for i, o in enumerate(order) if m.startswith(o)), 99))
    df = df.sort_values("order").drop(columns="order")
    df.to_csv(os.path.join(RESULTS, "model_comparison.csv"), index=False)
    md = df.to_markdown(index=False)
    print(md)
    with open(os.path.join(RESULTS, "model_comparison.md"), "w", encoding="utf-8") as f:
        f.write(md + "\n")


if __name__ == "__main__":
    main()
