# -*- coding: utf-8 -*-
"""Hospital-cohort external validation (XGB): MIMIC harmonized -> 185-case hospital cohort.
Simulated dry-run: proves the end-to-end pipeline is ready for real 90-day mRS data.
"""
from __future__ import annotations
import os, io
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, cohen_kappa_score, brier_score_loss
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings("ignore")

BASE = r"D:\TT paper\0811Temporal Pathway"
OUT = os.path.join(BASE, "results", "external_validation")
os.makedirs(OUT, exist_ok=True)
WORK = os.path.join(BASE, "work")

DROP = ["hadm_id", "subject_id", "discharge_location", "outcome_ordinal", "hospital_expire_flag"]

def load_mimic_static():
    s = pd.read_csv(os.path.join(BASE, "data", "processed", "mimic_stroke", "mimic_harmonized_static.csv"))
    s = s[s["outcome_ordinal"].notna()].copy()
    s.columns = [c.replace("lab_", "") if c.startswith("lab_") else c for c in s.columns]
    y = s["outcome_ordinal"].astype(int).values
    feats = [c for c in s.columns if c not in DROP]
    X = s[feats].astype(np.float32)
    X = X.loc[:, X.notna().any()]  # drop all-NaN columns
    return X, y

def load_hosp_static():
    s = pd.read_csv(os.path.join(WORK, "hospital_static.csv"))
    s.columns = [c.replace("lab_", "") if c.startswith("lab_") else c for c in s.columns]
    return s

Xm, ym = load_mimic_static()
h = load_hosp_static()
shared = [c for c in Xm.columns if c in h.columns]
print("[mimic] n =", len(ym), "features =", Xm.shape[1], "| shared =", len(shared))
Xm_s = Xm[shared].copy()
med = Xm_s.median()
Xm_s = Xm_s.fillna(med)
Xh = h[shared].astype(np.float32).fillna(med.to_dict())
yh = h["mrs90"].values.astype(float)
poor_h = (yh >= 3).astype(int)
y6 = np.minimum(yh, 5).astype(int)
print("[hospital] n =", len(h), "| good rate =", round((yh <= 2).mean(), 3),
      "| poor rate =", round(poor_h.mean(), 3))
print("mRS dist:", pd.Series(yh).value_counts().sort_index().to_dict())

def macro_auc_ovr(y_true, proba, classes):
    aucs = []
    for c in classes:
        if (y_true == c).sum() < 1:
            continue
        aucs.append(roc_auc_score((y_true == c).astype(int), proba[:, c]))
    return float(np.mean(aucs)) if aucs else float("nan")

def slope_int(p, y):
    lp = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    lr = LogisticRegression(max_iter=1000).fit(lp.reshape(-1, 1), y)
    return float(lr.coef_[0][0]), float(lr.intercept_[0])

rows = []
# ---- MIMIC internal 5-fold CV (binary) ----
skf = StratifiedKFold(5, shuffle=True, random_state=42)
aucs = []
oof_bin = np.zeros(len(ym))
for tr, te in skf.split(Xm_s, (ym >= 3).astype(int)):
    m = xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                          random_state=42, n_jobs=8)
    m.fit(Xm_s.iloc[tr], (ym[tr] >= 3).astype(int))
    p = m.predict_proba(Xm_s.iloc[te])[:, 1]
    oof_bin[te] = p
    aucs.append(roc_auc_score((ym[te] >= 3).astype(int), p))
rows.append(("mimic_cv_xgb_binary_auc", float(np.mean(aucs)), float(np.std(aucs)), ""))

# ---- MIMIC internal ordinal macro-AUC ----
oof6 = np.zeros((len(ym), 6), dtype=np.float32)
skf2 = StratifiedKFold(5, shuffle=True, random_state=42)
for tr, te in skf2.split(Xm_s, ym):
    m = xgb.XGBClassifier(objective="multi:softprob", num_class=6, n_estimators=300,
                          max_depth=4, learning_rate=0.05, subsample=0.8,
                          colsample_bytree=0.8, tree_method="hist", random_state=42, n_jobs=8)
    m.fit(Xm_s.iloc[tr], ym[tr])
    oof6[te] = m.predict_proba(Xm_s.iloc[te])
rows.append(("mimic_cv_xgb_ordinal_macroauc", macro_auc_ovr(ym, oof6, list(range(6))), float("nan"), ""))

# ---- Transfer: binary model ----
mbin = xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                         random_state=42, n_jobs=8)
mbin.fit(Xm_s, (ym >= 3).astype(int))
p_bin = mbin.predict_proba(Xh)[:, 1]
auc_bin = roc_auc_score(poor_h, p_bin)
s0, i0 = slope_int(p_bin, poor_h)
brier = brier_score_loss(poor_h, p_bin)
rows.append(("hospital_transfer_xgb_binary_auc", auc_bin, float("nan"), ""))
rows.append(("hospital_transfer_xgb_binary_brier", brier, float("nan"), ""))
rows.append(("hospital_transfer_xgb_binary_calib_slope_int", s0, i0, ""))

# ---- Transfer: ordinal model (6-class) ----
mord = xgb.XGBClassifier(objective="multi:softprob", num_class=6, n_estimators=300,
                         max_depth=4, learning_rate=0.05, subsample=0.8,
                         colsample_bytree=0.8, tree_method="hist", random_state=42, n_jobs=8)
mord.fit(Xm_s, ym)
proba6 = mord.predict_proba(Xh)
auc_ord = macro_auc_ovr(y6, proba6, list(range(6)))
acc_ord = accuracy_score(y6, proba6.argmax(1))
kap = cohen_kappa_score(y6, proba6.argmax(1))
# P(poor) from ordinal model
p_poor_ord = proba6[:, 3:].sum(1)
auc_poor_ord = roc_auc_score(poor_h, p_poor_ord)
mort_auc = roc_auc_score((yh == 6).astype(int), proba6[:, 5]) if (yh == 6).sum() >= 2 else float("nan")
rows.append(("hospital_transfer_xgb_ordinal_macroauc", auc_ord, float("nan"), ""))
rows.append(("hospital_transfer_xgb_ordinal_acc", acc_ord, float("nan"), ""))
rows.append(("hospital_transfer_xgb_ordinal_kappa", kap, float("nan"), ""))
rows.append(("hospital_transfer_xgb_ordinal_poor_auc", auc_poor_ord, float("nan"), ""))
rows.append(("hospital_transfer_xgb_mortality_auc", mort_auc, float("nan"), ""))
np.savez(os.path.join(OUT, "hospital_xgb_transfer.npz"),
         proba6=proba6, p_bin=p_bin, y6=y6, poor_h=poor_h, mrs90=yh,
         ivt=h["ivt"].values, mt=h["mt"].values, age=h["age"].values, nihss=h["nihss"].values,
         lost=h["lost"].values, gcs_first=h["gcs_first"].values)

# ---- Subgroups ----
def sub_auc(name, idx, yp_, p_):
    if idx.sum() < 10 or len(np.unique(yp_[idx])) < 2:
        return
    rows.append((name, roc_auc_score(yp_[idx], p_[idx]), int(idx.sum()), ""))
h["_p"] = p_bin
for g, lab in [(h["age"] < 65, "age<65"), ((h["age"] >= 65) & (h["age"] < 80), "age65-79"), (h["age"] >= 80, "age>=80")]:
    sub_auc("sub_age_" + lab, g.values, poor_h, p_bin)
for g, lab in [(h["nihss"] <= 5, "nihss0-5"), ((h["nihss"] >= 6) & (h["nihss"] <= 15), "nihss6-15"), (h["nihss"] >= 16, "nihss>=16")]:
    sub_auc("sub_nihss_" + lab, g.values, poor_h, p_bin)
for g, lab in [(h["ivt"] == 1, "ivt_yes"), (h["ivt"] == 0, "ivt_no")]:
    sub_auc("sub_" + lab, g.values, poor_h, p_bin)
for g, lab in [(h["mt"] == 1, "mt_yes"), (h["mt"] == 0, "mt_no")]:
    sub_auc("sub_" + lab, g.values, poor_h, p_bin)

res = pd.DataFrame(rows, columns=["metric", "value", "aux", "note"])
res.to_csv(os.path.join(OUT, "hospital_xgb_external_validation.csv"), index=False)
print(res.to_string(index=False))

# ---- decile calibration ----
df = pd.DataFrame({"p": p_bin, "y": poor_h})
df["dec"] = pd.qcut(df["p"], 10, duplicates="drop")
cal = df.groupby("dec", observed=True).agg(n=("y", "size"), obs=("y", "mean"), pred=("p", "mean")).reset_index()
print("\nDecile calibration:")
print(cal.round(3).to_string(index=False))

# ---- report ----
lines = []
lines.append("# Hospital-cohort external validation dry-run (185 simulated cases)")
lines.append("")
lines.append(f"- Template: `ENG Hospital_Cohort_Template 187 Simulated Examples.xlsx` (Patient_Info n = 185).")
lines.append(f"- MIMIC harmonized training cohort n = {len(ym)}, shared features = {len(shared)}.")
lines.append(f"- Outcome: 90-day mRS (0-6). Good rate {100*round((yh <= 2).mean(), 3):.1f}%; poor rate {100*round(poor_h.mean(), 3):.1f}%.")
lines.append("")
lines.append("## Results (XGB transfer)")
lines.append("```")
lines.append(res.to_string(index=False))
lines.append("```")
lines.append("")
lines.append("## Interpretation")
lines.append("- The dry-run proves the end-to-end pipeline (template -> harmonized schema -> model transfer) is ready.")
lines.append("- Metrics on simulated data are illustrative only; real hospital data will differ.")
lines.append(f"- 3 patients flagged lost-to-FU (SIM059/120/161) still have mRS filled; exclude or resolve in real data.")
io.open(os.path.join(WORK, "hospital_extval_report.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("\nreport written to work/hospital_extval_report.md")

