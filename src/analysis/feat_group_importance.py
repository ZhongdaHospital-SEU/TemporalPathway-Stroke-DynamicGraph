# -*- coding: utf-8 -*-
"""Feature-group importance for XGB static+TS (gain, grouped)."""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import xgboost as xgb

PROC = r"D:\TT paper\0811Temporal Pathway\data\processed\mimic_stroke"
RESULTS = r"D:\TT paper\0811Temporal Pathway\results"
N_CLASSES = 6

GROUPS = [
    ("GCS", ["gcs_eye", "gcs_motor", "gcs_verbal", "gcs_total"]),
    ("Motor exam", ["motor_l_arm", "motor_l_leg", "motor_r_arm", "motor_r_leg"]),
    ("Vitals", ["hr", "sbp", "dbp", "mbp", "rr", "spo2", "temp"]),
    ("Labs", ["lab_"]),
    ("Demographics", ["age", "gender", "race", "weight", "height"]),
    ("Admission", ["admission_type", "admission_location", "insurance", "marital", "ethnicity", "language"]),
    ("Comorbidity", ["charlson", "elixhauser", "cci_", "comorbidity"]),
    ("Other static", None),
]


def main():
    stat = pd.read_csv(os.path.join(PROC, "static_plus_ts.csv"))
    stat = stat[stat["outcome_ordinal"].notna()].copy()
    z = np.load(os.path.join(PROC, "icu_sequences.npz"))
    seq_hadms = set(int(k[4:]) for k in z.files if k.startswith("seq_"))
    stat = stat[stat["hadm_id"].isin(seq_hadms)].copy()
    drop = ["hadm_id", "subject_id", "discharge_location", "outcome_ordinal", "hospital_expire_flag"]
    X = stat.drop(columns=[c for c in drop if c in stat.columns])
    X = X.select_dtypes(include=[np.number])
    X = X.loc[:, X.median() == X.median()]
    X = X.fillna(X.median())
    y = stat["outcome_ordinal"].astype(int).values
    model = xgb.XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8,
                              objective="multi:softprob", num_class=N_CLASSES,
                              tree_method="hist", random_state=42, n_jobs=4)
    model.fit(X, y)
    imp = pd.Series(model.feature_importances_, index=X.columns)
    assigned = pd.Series("Other", index=X.columns)
    for name, keys in GROUPS:
        if keys is None:
            continue
        if name == "Labs":
            mask = X.columns.str.startswith("lab_")
        elif name in ("GCS", "Motor exam", "Vitals"):
            mask = X.columns.str.startswith(tuple(keys))
        else:
            mask = X.columns.str.contains("|".join(keys), case=False, regex=True)
        assigned[mask] = name
    # remaining static (not starting with feature names)
    for name, keys in GROUPS:
        if keys is None:
            continue
        for k in keys:
            pass
    g = imp.groupby(assigned).sum().sort_values(ascending=False)
    out = pd.DataFrame({"group": g.index, "gain": g.values})
    out.to_csv(os.path.join(RESULTS, "feat_group_importance.csv"), index=False)
    print(out.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
