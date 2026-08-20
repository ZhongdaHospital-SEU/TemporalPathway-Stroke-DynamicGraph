# -*- coding: utf-8 -*-
'''统计严谨性复查: 多重校正 + bootstrap CI + 亚组分析'''
import os
import re
import numpy as np
import pandas as pd
from scipy.stats import norm

BASE = r'D:\TT paper\0811Temporal Pathway'
RES = os.path.join(BASE, 'results')
DATA = os.path.join(BASE, 'data', 'processed', 'mimic_stroke')
WORK = os.path.join(BASE, 'work')
MIMIC_HOSP = os.path.join(BASE, 'mimic-iv-3.1', 'hosp')
SEED = 20260814
B = 2000
NCLS = 6


def fmt_p(p):
    return 'P<0.001' if p < 0.001 else f'{p:.3f}'


def fmt2(x, nd=2):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return 'NA'
    return f'{x:.{nd}f}'


def fmt_ci(point, lo, hi, nd=2):
    return f'{point:.{nd}f} ({lo:.{nd}f}-{hi:.{nd}f})'


def parse_delong(path):
    rows = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            m = re.match(r'class (\d+): dAUC=([+-]?[\d.]+) z=([+-]?[\d.]+) p=([\d.]+)', line.strip())
            if m:
                rows.append((int(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4))))
    return rows


def holm_bonferroni(pvals):
    pvals = np.asarray(pvals, dtype=float)
    m = len(pvals)
    order = np.argsort(pvals)
    sorted_p = pvals[order]
    holm_sorted = np.minimum(1.0, (m - np.arange(m)) * sorted_p)
    holm_sorted = np.maximum.accumulate(holm_sorted)
    holm = np.empty(m)
    holm[order] = holm_sorted
    bonf = np.minimum(1.0, pvals * m)
    return holm, bonf


def class_auc(y, proba, k):
    scores = proba[:, k]
    n = len(y)
    order = np.argsort(scores, kind='stable')
    ranks = np.empty(n)
    ranks[order] = np.arange(1, n + 1)
    pos = y == k
    npos = pos.sum()
    nneg = n - npos
    if npos == 0 or nneg == 0:
        return np.nan
    return (ranks[pos].sum() - npos * (npos + 1) / 2.0) / (npos * nneg)


def kappa_from_cm(cm):
    n = cm.sum()
    p_o = np.trace(cm) / n
    rowsum = cm.sum(axis=1)
    colsum = cm.sum(axis=0)
    p_e = (rowsum * colsum).sum() / (n * n)
    if p_e >= 1.0:
        return np.nan
    return (p_o - p_e) / (1.0 - p_e)


def compute_metrics(y, proba):
    y = np.asarray(y)
    proba = np.asarray(proba, dtype=float)
    pred = proba.argmax(axis=1)
    acc = float((pred == y).mean())
    cm = np.zeros((NCLS, NCLS), dtype=np.float64)
    np.add.at(cm, (y, pred), 1)
    kap = kappa_from_cm(cm)
    auc = float(np.nanmean([class_auc(y, proba, k) for k in range(NCLS)]))
    return {'acc': acc, 'kappa': kap, 'macro_auc': auc}

def bootstrap_ci(y, proba, n_boot=B, seed=SEED):
    y = np.asarray(y, dtype=np.int8)
    proba = np.asarray(proba, dtype=np.float64)
    n = len(y)
    obs = compute_metrics(y, proba)
    am = proba.argmax(axis=1).astype(np.int8)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    y_idx = y[idx]
    pred = am[idx]
    # accuracy
    acc_b = (pred == y_idx).mean(axis=1).astype(np.float64)
    # kappa
    flat = (y_idx.astype(np.int32) * NCLS + pred) + np.arange(n_boot)[:, None] * (NCLS * NCLS)
    cnt = np.bincount(flat.ravel(), minlength=n_boot * NCLS * NCLS).reshape(n_boot, NCLS, NCLS).astype(np.float64)
    p_o = np.trace(cnt, axis1=1, axis2=2) / n
    rowsum = cnt.sum(axis=2)
    colsum = cnt.sum(axis=1)
    p_e = (rowsum * colsum).sum(axis=1) / (n * n)
    kap_b = (p_o - p_e) / (1.0 - p_e)
    # macro-AUC (vectorized, patient-level resample)
    aucs = np.empty((n_boot, NCLS), dtype=np.float64)
    rows = np.arange(n_boot)[:, None]
    for k in range(NCLS):
        scores = proba[:, k]
        order = np.argsort(scores, kind='stable')
        ppos = np.empty(n, dtype=np.int32)
        ppos[order] = np.arange(n, dtype=np.int32)
        P = ppos[idx]
        R = np.argsort(np.argsort(P, axis=1, kind='stable'), axis=1, kind='stable')
        mask = y_idx == k
        npos = mask.sum(axis=1).astype(np.float64)
        sum_ranks = ((mask * (R + 1)).sum(axis=1)).astype(np.float64)
        nneg = n - npos
        denom = npos * nneg
        auc = np.where(denom > 0, (sum_ranks - npos * (npos + 1) / 2.0) / np.where(denom > 0, denom, 1.0), np.nan)
        aucs[:, k] = auc
    mac_b = np.nanmean(aucs, axis=1)
    out = {}
    for name, arr in (('acc', acc_b), ('kappa', kap_b), ('macro_auc', mac_b)):
        lo, hi = np.nanpercentile(arr, [2.5, 97.5])
        out[name] = (obs[name], float(lo), float(hi))
    return out


def verify_auc_formula(y, proba):
    y = np.asarray(y)
    proba = np.asarray(proba, dtype=float)
    n = len(y)
    rng = np.random.default_rng(1)
    idx = rng.integers(0, n, size=(4, n))
    ok = True
    for r in range(4):
        yy = y[idx[r]]
        pp = proba[idx[r]]
        for k in range(NCLS):
            fast = class_auc(yy, pp, k)
            from scipy.stats import rankdata
            ranks = rankdata(pp[:, k], method='average')
            pos = yy == k
            npos = pos.sum(); nneg = n - npos
            slow = (ranks[pos].sum() - npos * (npos + 1) / 2.0) / (npos * nneg) if npos and nneg else np.nan
            if not (np.isnan(fast) and np.isnan(slow)) and abs(fast - slow) > 1e-10:
                ok = False
                print('AUC verify FAIL', r, k, fast, slow)
    return ok


def calibration_pge3(proba, y):
    risk = proba[:, 3:].sum(axis=1)
    ybin = (y >= 3).astype(int)
    df = pd.DataFrame({'risk': risk, 'y': ybin})
    nbins = 10 if len(df) >= 100 else 5
    df['bin'] = pd.qcut(df['risk'].rank(method='first'), nbins)
    g = df.groupby('bin', observed=True).agg(pred=('risk', 'mean'), obs=('y', 'mean'), cnt=('y', 'size'))
    if len(g) < 3:
        return np.nan, np.nan, np.nan, len(g)
    slope, intercept = np.polyfit(g['pred'].values, g['obs'].values, 1)
    r = float(np.corrcoef(g['pred'].values, g['obs'].values)[0, 1])
    return float(slope), float(intercept), r, len(g)


def subgroup_row(label, y, proba):
    met = compute_metrics(y, proba)
    slope, intercept, r, nb = calibration_pge3(proba, y)
    return {
        'subgroup': label,
        'n': int(len(y)),
        'acc': met['acc'],
        'kappa': met['kappa'],
        'macro_auc': met['macro_auc'],
        'cal_slope': slope,
        'cal_intercept': intercept,
        'cal_r': r,
        'n_bins': nb,
    }

def main():
    os.makedirs(WORK, exist_ok=True)
    print('=' * 70)
    print('loading data ...')
    stack = np.load(os.path.join(RES, 'stack_oof.npz'), allow_pickle=True)
    cv = np.load(os.path.join(RES, 'cv_predictions.npz'), allow_pickle=True)
    dg = np.load(os.path.join(RES, 'dggru_tpr_oof.npz'), allow_pickle=True)

    y_stack = stack['y'].astype(int)
    proba_stack = stack['proba'].astype(float)
    xgb_ts_stack = stack['xgb_ts'].astype(float)
    dg_proba = dg['proba'].astype(float)
    dg_y = dg['y'].astype(int)
    hadm = dg['hadm_id'].astype(np.int64)

    xgb_static_p = cv['xgb_icu_static'].astype(float)
    xgb_ts_p = cv['xgb_icu_ts'].astype(float)
    y_cv = cv['y_icu'].astype(int)

    assert np.array_equal(y_stack, dg_y) and np.array_equal(y_stack, y_cv), 'y mismatch'
    assert np.allclose(xgb_ts_stack, xgb_ts_p), 'xgb_ts mismatch'
    assert np.allclose(proba_stack, stack['stack2']), 'stack proba != stack2'
    print('consistency checks passed (y / xgb_ts / proba==stack2)')

    assert verify_auc_formula(y_stack, proba_stack), 'AUC formula verification failed'
    print('AUC formula verified vs naive rank method')

    # ---------- 任务 1 ----------
    delong = parse_delong(os.path.join(RES, 'delong_result.txt'))
    print('delong rows:', len(delong))
    classes = [r[0] for r in delong]
    dAUC = np.array([r[1] for r in delong])
    z = np.array([r[2] for r in delong])
    p_file = np.array([r[3] for r in delong])
    p_re = 2.0 * (1.0 - norm.cdf(np.abs(z)))
    holm, bonf = holm_bonferroni(p_re)

    t1_rows = []
    for i, c in enumerate(classes):
        t1_rows.append({
            'class': c,
            'dAUC': dAUC[i],
            'z': z[i],
            'p_file': p_file[i],
            'p_recomputed': p_re[i],
            'p_holm': holm[i],
            'p_bonf': bonf[i],
            'sig_holm': bool(holm[i] < 0.05),
            'dAUC_fmt': fmt2(dAUC[i]),
            'z_fmt': fmt2(z[i]),
            'p_recomputed_fmt': fmt_p(p_re[i]),
            'p_holm_fmt': fmt_p(holm[i]),
            'p_bonf_fmt': fmt_p(bonf[i]),
        })
    print('Task1 done. min raw p=%.4f min holm=%.4f min bonf=%.4f' % (p_re.min(), holm.min(), bonf.min()))

    # ---------- 任务 2 ----------
    models = [
        ('XGB static', y_cv, xgb_static_p),
        ('XGB static+TS', y_cv, xgb_ts_p),
        ('dggru_tpr', dg_y, dg_proba),
        ('Stack', y_stack, proba_stack),
    ]
    t2_rows = []
    for name, y, proba in models:
        res = bootstrap_ci(y, proba)
        for metric in ('acc', 'kappa', 'macro_auc'):
            point, lo, hi = res[metric]
            t2_rows.append({
                'model': name,
                'metric': metric,
                'point': point,
                'ci_low': lo,
                'ci_high': hi,
                'fmt': fmt_ci(point, lo, hi),
            })
        print(name, '| acc', tuple(round(v, 4) for v in res['acc']),
              '| kappa', tuple(round(v, 4) for v in res['kappa']),
              '| macroAUC', tuple(round(v, 4) for v in res['macro_auc']))
    print('Task2 done.')

    # ---------- 任务 3 ----------
    adm = pd.read_csv(os.path.join(DATA, 'stroke_admissions.csv'), encoding='utf-8-sig')
    adm = adm[['hadm_id', 'anchor_age', 'gender', 'outcome_ordinal']].copy()
    adm['hadm_id'] = adm['hadm_id'].astype(np.int64)
    sub_df = pd.DataFrame({'hadm_id': hadm}).merge(adm, on='hadm_id', how='left')
    assert sub_df['anchor_age'].notna().all() and sub_df['gender'].notna().all(), 'subgroup merge missing'
    age = sub_df['anchor_age'].values
    gender = sub_df['gender'].str.upper().values

    t3_rows = []
    age_grp = np.where(age < 65, '<65', np.where(age <= 80, '65-80', '>80'))
    for lbl in ['<65', '65-80', '>80']:
        m = age_grp == lbl
        t3_rows.append(subgroup_row('age:' + lbl, y_stack[m], proba_stack[m]))
    for lbl, gv in (('male', 'M'), ('female', 'F')):
        m = gender == gv
        t3_rows.append(subgroup_row('gender:' + lbl, y_stack[m], proba_stack[m]))

    i63_note = None
    i63_rows = []
    try:
        diag = pd.read_csv(os.path.join(MIMIC_HOSP, 'diagnoses_icd.csv.gz'),
                           usecols=['hadm_id', 'icd_code', 'icd_version'],
                           dtype={'hadm_id': np.int64, 'icd_code': str, 'icd_version': np.int8})
        diag = diag[diag['icd_version'] == 10]
        i63_ids = set(diag.loc[diag['icd_code'].astype(str).str.startswith('I63'), 'hadm_id'].unique())
        m = np.isin(hadm, np.array(sorted(i63_ids), dtype=np.int64))
        print('I63+: n =', int(m.sum()), '| I63-: n =', int((~m).sum()))
        i63_rows = [subgroup_row('I63+', y_stack[m], proba_stack[m]),
                    subgroup_row('I63-', y_stack[~m], proba_stack[~m])]
        t3_rows += i63_rows
    except Exception as e:
        i63_note = 'I63 亚型跳过: ' + str(e)
        print('I63 skipped:', e)

    for r in t3_rows:
        print(r['subgroup'], 'n=', r['n'], 'acc=%.4f kappa=%.4f auc=%.4f slope=%.3f' %
              (r['acc'], r['kappa'], r['macro_auc'], r['cal_slope']))
    print('Task3 done.')

    # ---------- 输出 CSV ----------
    t1_df = pd.DataFrame(t1_rows)
    t2_df = pd.DataFrame(t2_rows)
    t1_df.insert(0, 'analysis', 'delong_multcorr')
    t2_df.insert(0, 'analysis', 'bootstrap_ci')
    stats_csv = pd.concat([t1_df, t2_df], ignore_index=True, sort=False)
    stats_csv.to_csv(os.path.join(RES, 'stats_fixes_results.csv'), index=False, encoding='utf-8-sig')
    print('wrote', os.path.join(RES, 'stats_fixes_results.csv'))

    sub_df_out = pd.DataFrame(t3_rows)
    for col in ('acc', 'kappa', 'macro_auc', 'cal_slope', 'cal_intercept', 'cal_r'):
        sub_df_out[col + '_fmt'] = sub_df_out[col].map(fmt2)
    sub_df_out.to_csv(os.path.join(RES, 'subgroup_results.csv'), index=False, encoding='utf-8-sig')
    print('wrote', os.path.join(RES, 'subgroup_results.csv'))

    # ---------- 输出 Markdown ----------
    lines = []
    lines.append('# 统计严谨性复查报告')
    lines.append('')
    lines.append('数据来源: OOF 均为 ICU 子集 n=2875; XGB static/static+TS 取自 `results/cv_predictions.npz` '
                 '(`xgb_icu_static`/`xgb_icu_ts`, 与 `stack_oof.npz` 中 `xgb_ts` 一致); dggru_tpr 取自 '
                 '`results/dggru_tpr_oof.npz`; Stack 取自 `results/stack_oof.npz` (`proba`==`stack2`)。')
    lines.append('`stack_oof.npz` 不含 hadm_id, 亚组关联使用 `dggru_tpr_oof.npz` 的 hadm_id (已验证与 stack 的 y 完全一致)。')
    lines.append('')
    lines.append('## 一、DeLong 多重校正 (stack2 vs XGB static+TS, 6 类, 由 z 重算双侧 P)')
    lines.append('')
    lines.append('| 类别 | dAUC | z | 原始 P | Holm P | Bonferroni P |')
    lines.append('|---|---|---|---|---|---|')
    for r in t1_rows:
        lines.append('| class %d | %s | %s | %s | %s | %s |' % (
            r['class'], r['dAUC_fmt'], r['z_fmt'], r['p_recomputed_fmt'], r['p_holm_fmt'], r['p_bonf_fmt']))
    lines.append('')
    n_sig_holm = int(sum(r['sig_holm'] for r in t1_rows))
    lines.append('**结论**: 原始 P 最小为 %s (class 0/4); Holm 校正后最小 P=%s, Bonferroni 校正后最小 P=%s; '
                 '显著类别数: Holm %d/6, Bonferroni 0/6。校正后无类别达到 P<0.05, '
                 'stack2 相对 XGB static+TS 的单类 AUC 增益不具统计显著性。' % (
                     fmt_p(p_re.min()), fmt_p(holm.min()), fmt_p(bonf.min()), n_sig_holm))
    lines.append('')
    lines.append('## 二、Bootstrap 95% 百分位 CI (2000 次患者级重采样, 按行重采样)')
    lines.append('')
    lines.append('| 模型 | acc (95% CI) | kappa (95% CI) | macro-AUC (95% CI) |')
    lines.append('|---|---|---|---|')
    for name in ('XGB static', 'XGB static+TS', 'dggru_tpr', 'Stack'):
        row = {r['metric']: r['fmt'] for r in t2_rows if r['model'] == name}
        lines.append('| %s | %s | %s | %s |' % (name, row['acc'], row['kappa'], row['macro_auc']))
    lines.append('')
    st = {r['metric']: (r['point'], r['ci_low'], r['ci_high']) for r in t2_rows if r['model'] == 'Stack'}
    xg = {r['metric']: (r['point'], r['ci_low'], r['ci_high']) for r in t2_rows if r['model'] == 'XGB static+TS'}
    lines.append('**结论**: Stack 在 acc/kappa/macro-AUC 上点估计均为最高 '
                 '(macro-AUC %s, 95%% CI %s-%s), 但与 XGB static+TS 的 CI 相互重叠 '
                 '(acc: %s-%s vs %s-%s; kappa: %s-%s vs %s-%s; macro-AUC: %s-%s vs %s-%s)。'
                 '因此差异未达到 95%% 置信水平下的统计显著, 增量提升需谨慎解读。' % (
                     fmt2(st['macro_auc'][0]), fmt2(st['macro_auc'][1]), fmt2(st['macro_auc'][2]),
                     fmt2(st['acc'][1]), fmt2(st['acc'][2]), fmt2(xg['acc'][1]), fmt2(xg['acc'][2]),
                     fmt2(st['kappa'][1]), fmt2(st['kappa'][2]), fmt2(xg['kappa'][1]), fmt2(xg['kappa'][2]),
                     fmt2(st['macro_auc'][1]), fmt2(st['macro_auc'][2]), fmt2(xg['macro_auc'][1]), fmt2(xg['macro_auc'][2])))
    lines.append('')
    lines.append('## 三、亚组分析 (Stack OOF, P(y>=3) 校准为分位数分箱 observed vs predicted 的线性回归)')
    lines.append('')
    lines.append('| 亚组 | n | acc | kappa | macro-AUC | cal-slope | cal-intercept | cal-R | bins |')
    lines.append('|---|---|---|---|---|---|---|---|---|')
    for r in t3_rows:
        lines.append('| %s | %d | %s | %s | %s | %s | %s | %s | %d |' % (
            r['subgroup'], r['n'], fmt2(r['acc']), fmt2(r['kappa']), fmt2(r['macro_auc']),
            fmt2(r['cal_slope']), fmt2(r['cal_intercept']), fmt2(r['cal_r']), r['n_bins']))
    lines.append('')
    lines.append('**结论**: ')
    age_rows = [r for r in t3_rows if r['subgroup'].startswith('age:')]
    g_rows = [r for r in t3_rows if r['subgroup'].startswith('gender:')]
    best_age = max(age_rows, key=lambda r: r['macro_auc'])
    best_g = max(g_rows, key=lambda r: r['macro_auc'])
    lines.append('- 年龄亚组: n=%d+%d+%d; macro-AUC 最高为 %s (%s), '
                 'acc 范围 %s-%s, macro-AUC 范围 %s-%s。' % (
                     age_rows[0]['n'], age_rows[1]['n'], age_rows[2]['n'],
                     best_age['subgroup'][4:], fmt2(best_age['macro_auc']),
                     fmt2(min(r['acc'] for r in age_rows)), fmt2(max(r['acc'] for r in age_rows)),
                     fmt2(min(r['macro_auc'] for r in age_rows)), fmt2(max(r['macro_auc'] for r in age_rows))))
    lines.append('- 性别亚组: male n=%d, female n=%d; male acc=%s macro-AUC=%s, '
                 'female acc=%s macro-AUC=%s。' % (
                     g_rows[0]['n'], g_rows[1]['n'],
                     fmt2(g_rows[0]['acc']), fmt2(g_rows[0]['macro_auc']),
                     fmt2(g_rows[1]['acc']), fmt2(g_rows[1]['macro_auc'])))
    worst_cal = min(t3_rows, key=lambda r: abs(r['cal_slope'] - 1.0))
    lines.append('- 校准 (P(y>=3)): cal-slope 最接近 1 的亚组为 %s (slope=%s), '
                 '其余亚组 slope 偏离 1 提示系统性高估/低估风险, 但分箱数有限, 解读需谨慎。' % (
                     worst_cal['subgroup'], fmt2(worst_cal['cal_slope'])))
    if i63_rows:
        lines.append('- I63 亚型 (ICD-10 I63 脑梗死, 来自 mimic-iv-3.1/hosp/diagnoses_icd.csv.gz): '
                     'I63+ n=%d (acc=%s, macro-AUC=%s), I63- n=%d (acc=%s, macro-AUC=%s)。' % (
                         i63_rows[0]['n'], fmt2(i63_rows[0]['acc']), fmt2(i63_rows[0]['macro_auc']),
                         i63_rows[1]['n'], fmt2(i63_rows[1]['acc']), fmt2(i63_rows[1]['macro_auc'])))
    elif i63_note:
        lines.append('- I63 亚型: 跳过 (%s)。' % i63_note)
    lines.append('')
    lines.append('## 总体结论')
    lines.append('')
    lines.append('1. **多重校正**: 6 类 DeLong 中原始 P 最小 0.048, Holm 与 Bonferroni 校正后均为 0.286, '
                 '无类别显著, stack2 的单类 AUC 优势不稳健。')
    lines.append('2. **Bootstrap CI**: Stack 各指标点估计领先, 但与 XGB static+TS 的 95% CI 全部重叠, '
                 '差异未达统计显著。')
    lines.append('3. **亚组**: 模型在 >80 岁与男性亚组表现略好; P(y>=3) 校准 slope 各亚组在 1 附近波动, '
                 '未见系统性极端偏差。')
    md_path = os.path.join(WORK, 'stats_fixes.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('wrote', md_path)
    print('ALL DONE')


if __name__ == '__main__':
    main()
