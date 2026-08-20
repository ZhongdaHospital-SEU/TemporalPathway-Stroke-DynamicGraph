# -*- coding: utf-8 -*-
"""Table 1: cohort baseline characteristics by ordinal discharge outcome."""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

PROC = r"D:\TT paper\0811Temporal Pathway\data\processed\mimic_stroke"
RESULTS = r"D:\TT paper\0811Temporal Pathway\results"
LABELS = ["HOME", "HOME CARE", "REHAB", "SNF", "HOSPICE", "DIED"]


def main():
    stat = pd.read_csv(os.path.join(PROC, "static_features.csv"))
    # age / gender from MIMIC-IV raw tables
    raw = r"D:\TT paper\0811Temporal Pathway\mimic-iv-3.1\hosp"
    try:
        adm = pd.read_csv(os.path.join(raw, "admissions.csv.gz"), usecols=["subject_id", "hadm_id", "admittime", "dischtime"])
        pat = pd.read_csv(os.path.join(raw, "patients.csv.gz"), usecols=["subject_id", "anchor_age", "gender", "anchor_year", "anchor_year_group"])
        adm["admittime"] = pd.to_datetime(adm["admittime"])
        pat["anchor_age"] = pd.to_numeric(pat["anchor_age"], errors="coerce")
        m = adm.merge(pat, on="subject_id", how="left")
        m["age"] = m["anchor_age"]
        stat = stat.merge(m[["hadm_id", "age", "gender"]], on="hadm_id", how="left")
    except Exception as e:
        print("age/gender merge failed:", e, flush=True)
    ts = pd.read_csv(os.path.join(PROC, "icu_timeseries_long.csv"), low_memory=False,
                     parse_dates=["charttime"])
    z = np.load(os.path.join(PROC, "icu_sequences.npz"))
    seq_hadms = set(int(k[4:]) for k in z.files if k.startswith("seq_"))
    stat = stat[stat["outcome_ordinal"].notna()].copy()
    stat["icu_seq"] = stat["hadm_id"].isin(seq_hadms).astype(int)

    # ICU length of stay (hours) and GCS stats from timeseries
    ts = ts[ts["hadm_id"].isin(stat["hadm_id"])]
    agg = ts.groupby("hadm_id").agg(
        los_h=("charttime", lambda s: (s.max() - s.min()).total_seconds() / 3600),
        n_obs=("charttime", "count"),
        gcs_first=("gcs_eye", "first"))
    # GCS first (eye+verbal+motor at first charted time)
    g = ts.sort_values("charttime").groupby("hadm_id")
    first = g.first()
    agg["gcs_total_first"] = first["gcs_eye"] + first["gcs_verbal"] + first["gcs_motor"]
    stat = stat.merge(agg, on="hadm_id", how="left")

    rows = []
    for k in range(6):
        sub = stat[stat["outcome_ordinal"] == k]
        rows.append({
            "outcome": LABELS[k], "n": len(sub),
            "age_mean": sub.get("age", pd.Series(dtype=float)).mean() if "age" in sub else np.nan,
            "male_pct": (sub["gender"] == "M").mean() if "gender" in sub else np.nan,
            "icu_seq_pct": sub["icu_seq"].mean(),
            "los_h_med": sub["los_h"].median(),
            "n_obs_med": sub["n_obs"].median(),
            "gcs_first_med": sub["gcs_total_first"].median(),
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "table1.csv"), index=False)
    print(df.to_string(index=False), flush=True)
    print(f"total: {len(stat)} (ICU-seq {stat['icu_seq'].sum()})", flush=True)


if __name__ == "__main__":
    main()
