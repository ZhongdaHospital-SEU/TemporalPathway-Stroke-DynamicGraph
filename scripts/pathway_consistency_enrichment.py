# -*- coding: utf-8 -*-
"""Cross-cohort direction consistency and stroke-process enrichment.

Task 1: test whether the direction of the 70 atlas pathways (GSE37587
subacute, FDR < 0.05) replicates in the other GEO cohorts.
Task 2: keyword-based classification of atlas pathways into stroke-biology
categories and hypergeometric enrichment tests vs the full atlas background.
"""
from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = r"D:\TT paper\0811Temporal Pathway"
DATA = os.path.join(ROOT, "data", "processed", "geo_pathway")
RES = os.path.join(ROOT, "results")
FIG = os.path.join(RES, "figures")
os.makedirs(FIG, exist_ok=True)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_atlas():
    atlas = pd.read_csv(os.path.join(DATA, "temporal_atlas.csv"))
    sig = atlas[
        (atlas["dataset"] == "GSE37587")
        & (atlas["phase"] == "subacute")
        & (atlas["fdr"] < 0.05)
    ].copy()
    sig = sig.sort_values("fdr").reset_index(drop=True)
    return atlas, sig


def load_cohort(gse):
    act = pd.read_csv(os.path.join(DATA, "activity_%s.csv" % gse), index_col=0)
    act.index = act.index.astype(str)
    act.columns = act.columns.astype(str)
    ph = pd.read_csv(os.path.join(DATA, "pheno_%s.csv" % gse))
    ph["geo_accession"] = ph["geo_accession"].astype(str)
    ph = ph.set_index("geo_accession")
    return act, ph


def dir_unpaired(act, labels):
    """labels: sample -> 'case'/'ctrl'. Return per-pathway t, p, dir (up = case mean > ctrl mean)."""
    case = [s for s in act.columns if labels.get(s) == "case"]
    ctrl = [s for s in act.columns if labels.get(s) == "ctrl"]
    rows = []
    for path in act.index:
        a = act.loc[path, case].to_numpy(dtype=float)
        b = act.loc[path, ctrl].to_numpy(dtype=float)
        t, p = st.ttest_ind(a, b, equal_var=False)
        d = "up" if a.mean() > b.mean() else "down"
        rows.append({"pathway": path, "t": t, "p": p, "dir": d})
    return pd.DataFrame(rows).set_index("pathway")


def dir_paired(act, ph, patient_col, time_col, t0, t1):
    """Paired t-test t1 vs t0 within patients. Return per-pathway t, p, dir."""
    samples = list(ph.index)
    pat = ph[patient_col].to_numpy()
    tim = ph[time_col].astype(str).to_numpy()
    t0s, t1s = str(t0), str(t1)
    rows = []
    for path in act.index:
        tmp = pd.DataFrame(
            {"patient": pat, "time": tim,
             "score": act.loc[path, samples].to_numpy(dtype=float)}
        )
        piv = tmp.pivot_table(index="patient", columns="time", values="score")
        piv = piv.reindex(columns=[t0s, t1s]).dropna()
        if len(piv) < 5:
            continue
        t, p = st.ttest_rel(piv[t1s], piv[t0s])
        d = "up" if piv[t1s].mean() > piv[t0s].mean() else "down"
        rows.append({"pathway": path, "t": t, "p": p, "dir": d})
    return pd.DataFrame(rows).set_index("pathway")


def make_comparisons():
    comparisons = []

    act, ph = load_cohort("GSE16561")
    labels = ph["group"].map(lambda g: "case" if g == "Stroke" else "ctrl")
    d = dir_unpaired(act, labels)
    comparisons.append({
        "name": "GSE16561_stroke_vs_control",
        "cohort": "GSE16561",
        "ctype": "disease_vs_control",
        "dir": d,
        "desc": ("Whole-blood stroke (n=39) vs control (n=24); cross-sectional "
                 "disease-state association, no time information available."),
    })

    act, ph = load_cohort("GSE22255")
    labels = ph["group"].map(lambda g: "case" if g == "Stroke" else "ctrl")
    d = dir_unpaired(act, labels)
    comparisons.append({
        "name": "GSE22255_stroke_vs_control",
        "cohort": "GSE22255",
        "ctype": "disease_vs_control",
        "dir": d,
        "desc": ("Whole-blood stroke (n=20) vs control (n=20), acute phase; "
                 "cross-sectional disease-state association."),
    })

    act, ph = load_cohort("GSE58294")
    for t in ["3", "5", "24"]:
        case_ids = ph[(ph["group"] == "Cardioembolic Stroke")
                      & (ph["time"].astype(str) == t)].index
        labels = pd.Series("ctrl", index=ph.index)
        labels.loc[case_ids] = "case"
        d = dir_unpaired(act, labels)
        comparisons.append({
            "name": "GSE58294_stroke_vs_control_%sh" % t,
            "cohort": "GSE58294",
            "ctype": "disease_vs_control",
            "dir": d,
            "desc": ("Cardioembolic stroke (n=23) vs control (n=23) at %sh; "
                     "cross-sectional disease-state association at an acute timepoint." % t),
        })

    stroke94 = ph[(ph["group"] == "Cardioembolic Stroke")
                  & (ph["time"].astype(str).isin(["3", "24"]))]
    d = dir_paired(act, stroke94, "patient", "time", "3", "24")
    comparisons.append({
        "name": "GSE58294_24h_vs_3h",
        "cohort": "GSE58294",
        "ctype": "temporal_paired",
        "dir": d,
        "desc": ("Paired 3h -> 24h within cardioembolic stroke patients (n=23); "
                 "direct temporal replication of the atlas acute-phase comparison."),
    })
    return comparisons


def bh_fdr(pvals):
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(p)
    out[order] = np.minimum(ranked, 1.0)
    return out


# ---------------------------------------------------------------------------
# Task 1: cross-cohort direction consistency
# ---------------------------------------------------------------------------
def consistency_analysis(atlas, sig, comparisons):
    per_path = sig[["pathway", "dir", "stat", "p", "fdr"]].copy()
    per_path.columns = ["pathway", "atlas_dir", "atlas_stat", "atlas_p", "atlas_fdr"]

    summary_rows = []
    for cmp_ in comparisons:
        name = cmp_["name"]
        d = cmp_["dir"].rename(columns={"t": "t_cohort", "p": "p_cohort",
                                        "dir": "dir_cohort"})[["t_cohort", "dir_cohort"]]

        m = sig.merge(d, left_on="pathway", right_index=True, how="left")
        m = m.dropna(subset=["dir_cohort"])
        n70 = int(len(m))
        conc70 = int((m["dir"] == m["dir_cohort"]).sum())
        frac70 = conc70 / n70 if n70 else float("nan")
        p_binom70 = st.binomtest(conc70, n70, p=0.5,
                                 alternative="two-sided").pvalue if n70 else float("nan")

        bg = atlas.merge(d, left_on="pathway", right_index=True, how="left")
        if name == "GSE58294_24h_vs_3h":
            bg = bg[bg["dataset"] != "GSE58294"]  # same comparison would be trivially concordant
        bg = bg.dropna(subset=["dir_cohort"])
        nb = int(len(bg))
        concb = int((bg["dir"] == bg["dir_cohort"]).sum())
        fracb = concb / nb if nb else float("nan")
        p_binomb = st.binomtest(concb, nb, p=0.5,
                                alternative="two-sided").pvalue if nb else float("nan")

        fisher_p = float("nan")
        if n70 and nb:
            table = [[conc70, n70 - conc70], [concb, nb - concb]]
            fisher_p = st.fisher_exact(table, alternative="two-sided").pvalue

        summary_rows.append({
            "comparison": name,
            "cohort": cmp_["cohort"],
            "type": cmp_["ctype"],
            "description": cmp_["desc"],
            "n_70": n70,
            "concordant_70": conc70,
            "fraction_70": frac70,
            "binom_p_70": p_binom70,
            "n_background": nb,
            "concordant_background": concb,
            "fraction_background": fracb,
            "binom_p_background": p_binomb,
            "fisher_p_70_vs_background": fisher_p,
        })

        per_path[name + "_t"] = per_path["pathway"].map(d["t_cohort"])
        per_path[name + "_dir"] = per_path["pathway"].map(d["dir_cohort"])
        conc = per_path["atlas_dir"] == per_path[name + "_dir"]
        per_path[name + "_concordant"] = conc.where(per_path[name + "_dir"].notna())

    return pd.DataFrame(summary_rows), per_path


# ---------------------------------------------------------------------------
# Task 2: stroke-process classification and enrichment
# ---------------------------------------------------------------------------
CATEGORY_PATTERNS = [
    ("coagulation", [
        "coagulation", "hemostasis", "haemostasis", "hemostatic", "platelet",
        "thromb*", "fibrin*", "plasmin*", "clot*", "von willebrand",
    ]),
    ("hypoxia/ischemia", [
        "hypox*", "ischemi*", "ischaemi*", "hif-*", "hif1*", "oxygen*",
        "reoxygen*", "reperfusion", "angiogen*", "neovascular*",
        "vascular endothelial growth", "vegf", "vegfa", "vegfr*",
        "oxidative stress", "reactive oxygen",
    ]),
    ("inflammation/immune", [
        "inflammat*", "immun*", "cytokine", "chemokine", "interleuk*",
        "il-1", "il-2", "il-3", "il-4", "il-5", "il-6", "il-7", "il-8",
        "il-9", "il-10", "il-11", "il-12", "il-13", "il-15", "il-16",
        "il-17", "il-18", "il-21", "il-22", "il-23", "il-24", "il-33",
        "tumor necrosis factor", "tumour necrosis factor", "tnf", "nf-kappa",
        "nf-kb", "toll-like", "toll like", "tlr*", "complement", "interferon",
        "ifn-*", "leukocyte", "leucocyte", "neutrophil", "macrophag*",
        "monocyt*", "mast cell", "eosinophil", "basophil", "dendritic cell",
        "natural killer", "nk cell", "lymphocyt*", "t cell", "b cell",
        "antigen", "pattern recognition", "fc gamma", "fc epsilon",
        "c-type lectin", "rig-i", "nod-like", "cytosolic dna-sensing",
        "cytokine-cytokine", "hematopoiesis", "haematopoiesis",
        "hematopoietic*", "haematopoietic*", "graft*", "allograft",
        "autoimmune", "asthma", "rheumatoid", "inflammatory bowel",
    ]),
    ("neurogenesis/plasticity", [
        "neuro*", "neural", "neuron*", "synapse", "synaptic", "synapto*",
        "axon", "axonal", "axon guidance", "dendrit*", "neurit*",
        "long-term potentiation", "long term potentiation", "gaba*",
        "glutamatergic*", "glutamate receptor", "dopaminerg*",
        "dopamine receptor", "serotonerg*", "serotonin receptor",
        "cholinerg*", "acetylcholin*", "nicotine", "morphine", "cocaine",
        "amphetamine", "opioid", "endocannabinoid", "circadian",
        "phototransduction", "olfactory", "taste", "behavior", "behaviour",
        "learning", "memory", "cognition", "cognitive", "brain", "cerebral",
        "neuroactive ligand", "nerve", "action potential", "nervous system",
    ]),
    ("metabolism", [
        "metabol*", "glycolys*", "gluconeogenes*", "pentose phosphate",
        "fatty acid", "lipid", "cholesterol", "steroid", "bile", "amino acid",
        "urea cycle", "oxidative phosphorylation", "citrate", "tca cycle",
        "carbon metabolism", "nitrogen", "sulfur metabolism", "drug metabolism",
        "cytochrome p450", "xenobiotic", "insulin", "glucagon", "ppar*",
        "ampk", "adipocytokine", "biosynthesis", "catabolic", "anabolic",
        "coenzyme", "vitamin", "ketone", "glycan", "glycosaminoglycan",
        "inositol phosphate", "phosphatidylinositol", "sphingolipid",
        "glycerophospholipid", "glycerolipid", "ether lipid", "arachidonic",
        "linoleic", "butanoate", "propanoate", "beta-alanine", "taurine",
        "carnitine", "atp synthesis", "acetyl-coa", "acyl-coa", "carbohydrate",
        "glucose", "glycogen", "sugar", "pyruvate", "succinate", "malate",
        "oxoglutarate", "electron transport", "energy",
    ]),
    ("apoptosis", [
        "apoptos*", "programmed cell death", "cell death", "necroptos*",
        "pyroptos*", "ferroptos*", "death receptor", "p53*", "tp53",
        "caspas*", "bcl-2", "bcl2",
    ]),
]

CATEGORY_LABELS = {
    "coagulation": "Coagulation / hemostasis",
    "hypoxia/ischemia": "Hypoxia / ischemia",
    "inflammation/immune": "Inflammation / immune",
    "neurogenesis/plasticity": "Neurogenesis / plasticity",
    "metabolism": "Metabolism",
    "apoptosis": "Apoptosis / cell death",
    "other": "Other",
}

CATEGORY_ORDER = ["inflammation/immune", "hypoxia/ischemia", "coagulation",
                  "neurogenesis/plasticity", "metabolism", "apoptosis", "other"]


def _to_regex(p):
    """Convert a keyword to regex. Trailing '*' marks a prefix match."""
    prefix = p.endswith("*")
    core = p[:-1] if prefix else p
    s = re.escape(core)
    if core[0].isalnum() or core[0] == "_":
        s = r"\b" + s
    if prefix:
        s = s + r"\w*"
    elif core[-1].isalnum() or core[-1] == "_":
        s = s + r"s?\b"
    return s


_CATEGORY_RE = [
    (cat, re.compile("|".join(_to_regex(p) for p in pats), re.IGNORECASE))
    for cat, pats in CATEGORY_PATTERNS
]


def classify(name):
    n = name.split(":", 1)[-1]
    for cat, rx in _CATEGORY_RE:
        if rx.search(n):
            return cat
    return "other"


def enrichment_analysis(atlas, sig):
    atlas = atlas.copy()
    sig = sig.copy()
    atlas["category"] = atlas["pathway"].map(classify)
    sig["category"] = sig["pathway"].map(classify)
    N = int(len(atlas))
    n = int(len(sig))

    rows = []
    for cat in CATEGORY_ORDER:
        K = int((atlas["category"] == cat).sum())
        k = int((sig["category"] == cat).sum())
        p_hyper = st.hypergeom.sf(k - 1, N, K, n)  # P(X >= k), right tail
        a = k + 0.5
        b = (n - k) + 0.5
        c = (K - k) + 0.5
        d = (N - n - K + k) + 0.5
        orr = (a * d) / (b * c)
        se = np.sqrt(1.0 / a + 1.0 / b + 1.0 / c + 1.0 / d)
        ci_lo = np.exp(np.log(orr) - 1.96 * se)
        ci_hi = np.exp(np.log(orr) + 1.96 * se)
        fold = (k / n) / (K / N) if K else float("nan")
        examples = "; ".join(sig.loc[sig["category"] == cat, "pathway"].tolist())
        rows.append({
            "category": cat,
            "label": CATEGORY_LABELS[cat],
            "background_n": K,
            "background_prop": K / N,
            "significant_n": k,
            "significant_prop": k / n,
            "expected_n": n * K / N,
            "fold_enrichment": fold,
            "odds_ratio": orr,
            "ci_low": ci_lo,
            "ci_high": ci_hi,
            "p_hyper": p_hyper,
            "sig_pathways": examples,
        })
    enrich = pd.DataFrame(rows)
    enrich["fdr_bh"] = bh_fdr(enrich["p_hyper"].to_numpy())
    return enrich


# ---------------------------------------------------------------------------
# Figure: enrichment odds-ratio forest plot
# ---------------------------------------------------------------------------
def make_figure(enrich):
    plt.rcParams.update({"font.size": 10, "svg.fonttype": "none",
                         "font.family": "DejaVu Sans"})
    order = enrich.sort_values("odds_ratio")["category"].tolist()

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    for i, cat in enumerate(order):
        r = enrich[enrich["category"] == cat].iloc[0]
        color = "#C0392B" if r["fdr_bh"] < 0.05 else "#5D6D7E"
        ax.plot([r["ci_low"], r["ci_high"]], [i, i], color=color, lw=2.0, zorder=3)
        ax.scatter([r["odds_ratio"]], [i], color=color, s=70, zorder=4,
                   edgecolor="white", linewidth=0.8)
        star = "*" if r["fdr_bh"] < 0.05 else ""
        ax.text(r["ci_high"] * 1.35, i, star, va="center", ha="left",
                color=color, fontsize=14, zorder=5)

    ax.axvline(1.0, color="#222222", lw=1.0, ls="--", zorder=2)
    ax.set_yticks(range(len(order)))
    labels = ["%s (%d/%d)" % (CATEGORY_LABELS[c],
                              int(enrich.loc[enrich["category"] == c, "significant_n"].iloc[0]),
                              int(enrich.loc[enrich["category"] == c, "background_n"].iloc[0]))
              for c in order]
    ax.set_yticklabels(labels)
    ax.set_xscale("log")
    lo_all = enrich["ci_low"].min()
    hi_all = enrich["ci_high"].max()
    ax.set_xlim(lo_all * 0.6, hi_all * 3.0)
    ax.set_xlabel("Odds ratio (95% CI, log scale)")
    ax.set_title("Stroke-process enrichment in the 70-pathway atlas")
    ax.text(0.01, 0.02, "Counts: significant / background\n* BH-FDR < 0.05",
            transform=ax.transAxes, fontsize=8, va="bottom",
            color="#444444")
    fig.tight_layout()

    svg_path = os.path.join(FIG, "panel_pathway_enrich.svg")
    png_path = os.path.join(FIG, "panel_pathway_enrich.png")
    fig.savefig(svg_path, format="svg")
    fig.savefig(png_path, format="png", dpi=300)
    plt.close(fig)
    return svg_path, png_path


# ---------------------------------------------------------------------------
# Text summary
# ---------------------------------------------------------------------------
def write_stats(summary, enrich, atlas, sig, svg_path, png_path):
    L = []
    L.append("PATHWAY CROSS-COHORT CONSISTENCY AND STROKE-PROCESS ENRICHMENT")
    L.append("=" * 78)
    L.append("")
    L.append("Atlas: %d significant pathways (GSE37587 subacute, paired" % len(sig))
    L.append("       Baseline -> Follow-Up within n=34 stroke patients, FDR < 0.05).")
    L.append("Background: all %d atlas tests (GSE37587 subacute n=6308 +" % len(atlas))
    L.append("            GSE58294 acute n=6306, paired 3h -> 24h within n=23 patients).")
    L.append("")
    L.append("TASK 1 - CROSS-COHORT DIRECTION CONSISTENCY")
    L.append("-" * 78)
    L.append("")
    L.append("What each cohort supports (no overclaiming):")
    for _, r in summary.iterrows():
        L.append("  - %-38s [%s] %s" % (r["comparison"], r["type"], r["description"]))
    L.append("  - GSE37587 is the atlas cohort itself (the 70 were selected there);")
    L.append("    it is not used as a replication cohort.")
    L.append("")
    L.append("Concordance = fraction of atlas pathways whose direction (up/down)")
    L.append("matches the direction computed in the comparison cohort.")
    L.append("Binomial sign test: two-sided exact test vs P = 0.5.")
    L.append("")
    hdr = "%-38s %5s %5s %8s %10s | %8s %8s %10s %12s | %10s" % (
        "comparison", "n70", "conc", "frac70", "binomP70",
        "nBg", "concBg", "fracBg", "binomPBg", "fisherP")
    L.append(hdr)
    L.append("-" * len(hdr))
    for _, r in summary.iterrows():
        L.append("%-38s %5d %5d %8.3f %10.2e | %8d %8d %10.3f %12.2e | %10.2e" % (
            r["comparison"][:38], r["n_70"], r["concordant_70"], r["fraction_70"],
            r["binom_p_70"], r["n_background"], r["concordant_background"],
            r["fraction_background"], r["binom_p_background"],
            r["fisher_p_70_vs_background"]))
    L.append("")
    L.append("Notes:")
    L.append("  - Background for GSE58294_24h_vs_3h excludes the 6,306 GSE58294 acute")
    L.append("    atlas rows (the same paired comparison would be trivially concordant),")
    L.append("    leaving the 6,308 GSE37587 rows as the independent background; 6,293 of")
    L.append("    these had computable directions in GSE58294 (15 lacked ssGSEA activity).")
    L.append("  - For disease-vs-control comparisons, 'up' means higher in stroke than")
    L.append("    control; for temporal comparisons 'up' means higher at the later time.")
    L.append("  - fisherP = two-sided Fisher exact test comparing concordance among the")
    L.append("    70 atlas pathways vs the background concordance rate.")
    L.append("")
    L.append("TASK 2 - STROKE-PROCESS ENRICHMENT")
    L.append("-" * 78)
    L.append("")
    L.append("Classification: pathway-name keyword regex, single assignment by priority")
    L.append("(coagulation > hypoxia/ischemia > inflammation/immune > neurogenesis/")
    L.append("plasticity > metabolism > apoptosis > other).")
    L.append("Test: hypergeometric enrichment (right tail), selected n=%d of N=%d;" % (len(sig), len(atlas)))
    L.append("BH-FDR across %d categories; OR with Haldane-Anscombe 0.5 correction." % len(enrich))
    L.append("")
    L.append("%-24s %6s %8s %6s %8s %8s %8s %10s %10s %10s %10s" % (
        "category", "bgN", "bgProp", "sigN", "sigProp", "fold", "OR", "CIlow", "CIhigh", "P", "FDR"))
    for _, r in enrich.iterrows():
        L.append("%-24s %6d %8.4f %6d %8.4f %8.2f %8.2f %10.2f %10.2f %10.2e %10.2e" % (
            r["category"][:24], r["background_n"], r["background_prop"],
            r["significant_n"], r["significant_prop"], r["fold_enrichment"],
            r["odds_ratio"], r["ci_low"], r["ci_high"], r["p_hyper"], r["fdr_bh"]))
    L.append("")
    L.append("Examples among the 70 significant pathways per category:")
    for _, r in enrich.iterrows():
        if r["significant_n"] > 0:
            L.append("  %-24s %s" % (r["category"] + ":", r["sig_pathways"]))
    L.append("")
    L.append("Outputs:")
    L.append("  results/pathway_consistency.csv")
    L.append("  results/pathway_process_enrichment.csv")
    L.append("  %s" % svg_path)
    L.append("  %s" % png_path)
    txt = "\n".join(L) + "\n"
    with open(os.path.join(RES, "pathway_consistency_stats.txt"), "w",
              encoding="utf-8") as f:
        f.write(txt)
    return txt


def main():
    atlas, sig = load_atlas()
    print("[load] atlas rows: %d; significant atlas pathways: %d" % (len(atlas), len(sig)))

    comparisons = make_comparisons()
    summary, per_path = consistency_analysis(atlas, sig, comparisons)
    per_path.to_csv(os.path.join(RES, "pathway_consistency.csv"), index=False)
    print("[task1] per-pathway CSV written: %d rows" % len(per_path))

    enrich = enrichment_analysis(atlas, sig)
    enrich.to_csv(os.path.join(RES, "pathway_process_enrichment.csv"), index=False)
    print("[task2] enrichment CSV written: %d categories" % len(enrich))

    svg_path, png_path = make_figure(enrich)
    print("[fig] %s" % svg_path)
    print("[fig] %s" % png_path)

    txt = write_stats(summary, enrich, atlas, sig, svg_path, png_path)
    print(txt)
    print("[done] all outputs written")


if __name__ == "__main__":
    main()
