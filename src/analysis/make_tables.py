# -*- coding: utf-8 -*-
"""Final publication tables: Table 1-5 (CSV + Markdown).

Number format rules:
- general values: 2 decimals
- p-values: P>=0.001 keep 3 decimals; P<0.001 written as "P<0.001"
"""
from __future__ import annotations
import os
import re
import numpy as np
import pandas as pd

BASE = r"D:\TT paper\0811Temporal Pathway"
RES = os.path.join(BASE, "results")
OUT = os.path.join(RES, "tables")
os.makedirs(OUT, exist_ok=True)
LABELS = ["HOME", "HOME CARE", "REHAB", "SNF", "HOSPICE", "DIED"]


def fmt2(x):
    try:
        return f"{float(x):.2f}"
    except (TypeError, ValueError):
        return ""


def fmt3(x):
    try:
        return f"{float(x):.3f}"
    except (TypeError, ValueError):
        return ""


def fmt_p(p):
    p = float(p)
    if p < 0.001:
        return "P<0.001"
    return f"{p:.3f}"


def write(name, df, index=False):
    df.to_csv(os.path.join(OUT, name + ".csv"), index=index)
    md = df.to_markdown(index=index)
    with open(os.path.join(OUT, name + ".md"), "w", encoding="utf-8") as f:
        f.write(md + chr(10))
    print("saved " + name, flush=True)
    print(md, flush=True)


def table1():
    t = pd.read_csv(os.path.join(RES, "table1.csv"))
    t["age_mean"] = t["age_mean"].map(fmt2)
    t["male_pct"] = (t["male_pct"].astype(float) * 100).round(1).map(lambda x: f"{x:.1f}%")
    t["icu_seq_pct"] = (t["icu_seq_pct"].astype(float) * 100).round(1).map(lambda x: f"{x:.1f}%")
    t["los_h_med"] = t["los_h_med"].map(fmt2)
    t["n_obs_med"] = t["n_obs_med"].map(lambda x: f"{int(x)}")
    t["gcs_first_med"] = t["gcs_first_med"].map(lambda x: f"{int(x)}")
    t.columns = ["Outcome", "n", "Age (mean)", "Male (%)", "ICU-seq (%)",
                 "LOS (h, median)", "Obs (median)", "First GCS (median)"]
    write("Table1", t)


def table2():
    rows = []
    base = open(os.path.join(RES, "cv_baselines.txt"), encoding="utf-8").read()
    for line in base.splitlines():
        m = re.match(r"XGB (.+?) \| acc: ([\d.]+) \| kappa: ([\d.]+) \| auc: ([\d.]+)", line)
        if m:
            rows.append({"Model": "XGB " + m.group(1).strip(), "acc": float(m.group(2)),
                         "kappa": float(m.group(3)), "macro-AUC": float(m.group(4))})
    for fn in ["gru_seqonly", "gru", "gru_tpr", "gru_tsstatic", "dggru", "dggru_tpr", "gru_long"]:
        p = os.path.join(RES, fn + "_result.txt")
        if not os.path.exists(p):
            continue
        t = open(p, encoding="utf-8").read()
        m = re.match(r"[\w+-]+ \| acc: ([\d.]+)\+/-[\d.]+ \| kappa: ([\d.]+) \| auc: ([\d.]+)", t.replace(" ordinal |", " |"))
        if m:
            name = {"gru_seqonly": "GRU seq-only", "gru": "GRU+static", "gru_tpr": "GRU+TPR",
                    "gru_tsstatic": "GRU+static+TS", "dggru": "DG-GRU", "dggru_tpr": "DG-GRU+TPR",
                    "gru_long": "GRU long (60ep)"}[fn]
            rows.append({"Model": name, "acc": float(m.group(1)),
                         "kappa": float(m.group(2)), "macro-AUC": float(m.group(3))})
    st = open(os.path.join(RES, "stacking_result.txt"), encoding="utf-8").read()
    m = re.search(r"stack_xgb_ts\+xgb_s\+dggru_tpr\s+acc=([\d.]+) kappa=([\d.]+) macroAUC=([\d.]+)", st)
    if m:
        rows.append({"Model": "Stack (proposed)", "acc": float(m.group(1)),
                     "kappa": float(m.group(2)), "macro-AUC": float(m.group(3))})
    df = pd.DataFrame(rows)
    df["acc"] = df["acc"].map(fmt2); df["kappa"] = df["kappa"].map(fmt2)
    df["macro-AUC"] = df["macro-AUC"].map(fmt3)
    write("Table2", df)


def table3():
    rows = [
        {"Model": "GRU", "AUC": 0.8131, "dAUC(GRU+TPR)": 0.0017},
        {"Model": "GRU+TPR", "AUC": 0.8148, "dAUC(GRU+TPR)": "-"},
        {"Model": "DG-GRU", "AUC": 0.8132, "dAUC(DG-GRU+TPR)": 0.0009},
        {"Model": "DG-GRU+TPR", "AUC": 0.8141, "dAUC(DG-GRU+TPR)": "-"},
    ]
    df = pd.DataFrame(rows)
    df["AUC"] = df["AUC"].map(fmt3)
    for c in ["dAUC(GRU+TPR)", "dAUC(DG-GRU+TPR)"]:
        df[c] = df[c].map(lambda x: "-" if x == "-" else fmt3(x))
    write("Table3", df)


def table4():
    txt = open(os.path.join(RES, "clinical_eval.txt"), encoding="utf-8").read()
    lines = txt.splitlines()
    rows = []
    for ln in lines:
        m = re.search(r"cutoff y>=([1-5]): category-free NRI=([+-][\d.]+)\s+IDI=([+-][\d.]+)", ln)
        if m and "ICU timeseries" in txt:
            # only the ICU-timeseries block
            if "ICU timeseries" in " ".join(lines[:lines.index(ln)]):
                rows.append({"Cutoff": "y>=" + m.group(1), "NRI": m.group(2), "IDI": m.group(3)})
    # simpler: parse by section marker
    rows = []
    sec = None
    for ln in lines:
        if ln.startswith("=== Incremental value:"):
            sec = ln.replace("=== Incremental value:", "").replace("===", "").strip()
        m = re.search(r"cutoff y>=([1-5]): category-free NRI=([+-][\d.]+)\s+IDI=([+-][\d.]+)", ln)
        if m and sec:
            rows.append({"Increment": sec, "Cutoff": "y>=" + m.group(1),
                         "NRI": fmt3(float(m.group(2))), "IDI": fmt3(float(m.group(3)))})
    df = pd.DataFrame(rows)
    write("Table4", df)


def table5():
    g = pd.read_csv(os.path.join(RES, "feat_group_importance.csv"))
    g["gain"] = g["gain"].map(fmt2)
    g.columns = ["Feature group", "Gain"]
    write("Table5", g)


if __name__ == "__main__":
    table1()
    table2()
    table3()
    table4()
    table5()
