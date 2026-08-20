# 补强方案总结报告（证据链与逻辑严密性）

- 日期：2026-08-15；状态：8 项补强分析全部完成并落盘（子代理 + 主线程接管）
- 目标期刊：CSBJ（已定）；写作阶段待用户确认后启动，正文用 SCI 地道英文

## 一、主结果（不变）
- Stack（XGB static + XGB static+TS + DG-GRU+TPR 三级级联，5 折 OOF）：acc 0.5391 / kappa 0.4073 / macro-AUC 0.8304
- Stack vs XGB static+TS：bootstrap ΔAUC 95%CI [+0.0005, +0.0084]；DeLong class0 P=0.048、class4 P=0.047

## 二、8 项补强分析结论

### 1. 超参数敏感性：TPR 正则强度 λ 网格扫描
- 文件：`results/lambda_sweep_results_bootstrap.csv`、`work/lambda_sweep_final_report.md`
- λ1∈{0.05,0.1,0.2,0.5,1.0,2.0} × λ2∈{0.01,0.05,0.1}（+0.2/0.2）= 18 配置，折均值 macro-AUC 0.8140–0.8172（极差 0.0032）
- bootstrap 1000 次患者级重采样：默认点 (0.2,0.05) ΔAUC vs 无 TPR 基线 = +0.0009（95%CI −0.0006~+0.0024）；18 个中仅 6 个 CI 下限 >0
- 结论：TPR 正则强度对性能几乎无影响，单调先验强度 λ1 增大有微正趋势（+0.003/40 倍），但远低于临床意义阈值。

### 2. 负对照实验（TPR 因果性）
- 文件：`results/dggru_tpr_anti_oof.npz`、`results/dggru_tpr_perm_oof.npz`、`work/negctrl_report.md`
- anti（反单调约束）0.8146、perm（通路成员置换）0.8152 vs 主模型 0.8147、无 TPR 基线 0.8136
- 结论：破坏通路先验不降低性能 → TPR 的判别贡献≈0，三方一致（λ 扫描 + 负对照 + bootstrap CI）。

### 3. 时间验证与时间窗消融
- 文件：`results/temporal_validation_results.csv`、`work/temporal_validation_report.md`
- 时间切分（train ≤2019 → test 2020–22）：static 0.8094 / static+TS 0.8114（ΔAUC +0.002，小于 OOF 的 +0.0087）；
  扩展窗口外推 2017–19：0.8195→0.8222（+0.003）；2011–13 小样本持平。
- 窗口消融：全住院期 0.8235 > 48h 0.7944 ≈ 24h 0.7953 > 6h 0.7870 → 缩短窗口性能下降，无时间泄漏迹象。
- 注意：跨时间外推校准偏差显著（calib P<0.001），Limitations 需如实报告。

### 4. 统计校正与亚组分析
- 文件：`work/stats_fixes.md`、`results/stats_fixes_results.csv`、`results/subgroup_results.csv`
- DeLong 6 类多重校正：原始 P 最小 0.048（class 0/4）→ Holm/Bonferroni 均 0.286，校正后无显著类别。
- Bootstrap 2000 次：Stack 点估计领先但与 XGB static+TS 的 95%CI 全部重叠 → 差异未达统计显著，增量解读需谨慎。
- 亚组：age <65 0.83 / 65–80 0.84 / >80 0.78；male 0.84 / female 0.82；校准 slope 0.98–1.14。

### 5. 通路检验稳健性与免疫细胞组成敏感性
- 文件：`results/pathway_sensitivity_results.csv`、`work/pathway_sensitivity.md`
- 三套检验（t+BH / Wilcoxon+BH / t+1000 次置换经验 FDR）：GSE37587 显著通路 70/90/79，与 atlas-70 交集 ≥62；GSE58294 均 0（原结论稳健）。
- atlas 复现逐条一致（max|ΔP|=6.66e-16）。
- 免疫细胞校正：中性粒校正后 atlas-70 中 70/70 仍显著（dropped 0）；ANCOVA 61/70；7 类细胞联合校正 70/70；通路变化与中性粒变化中位 |r|=0.14。
- 结论：70 条显著通路对检验方法与细胞组成校正高度稳健，不能被归因为白细胞比例变化。

### 6. 标签鲁棒性
- 文件：`results/label_robustness_results.csv`、`work/label_robustness.md`
- 4 种标签定义（原 6 类 + 3 种相邻类合并）下 static+TS 均优于 static（Δ=+0.006~+0.010），TS 增益方向完全稳健；对照精确复现 cv_baselines.txt。

### 7. 随机种子稳健性
- 文件：`results/seed_robustness_results.csv`、`work/seed_robustness.md`
- 5 seeds × 5 折：XGB-ts macro-AUC 0.8255–0.8284（mean 0.8270±0.0013），Stack 0.8292–0.8321（mean 0.8307±0.0013）；Stack>XGB-ts 在 5/5 种子成立。

### 8. 信息泄漏审计
- 文件：`work/leak_audit_report.md`
- 三个训练脚本均丢弃 discharge_location/outcome_ordinal/hospital_expire_flag；正对照（discharge_location 作特征）macro-AUC=1.0000 → 审计灵敏、实际无泄漏。
- 序列锚定 intime（首记录中位 +0.07h）；长表 29% 越界行为 349 例多次 ICU 停留的比对伪影，非坏数据；24h 窗口消融性能下降支持无时间泄漏。

## 三、关键决策点
- **叙事转向（已确认）**：TPR 通路先验对判别性能的贡献≈0（λ 扫描、负对照、bootstrap CI 三方一致）。
  正文将 TPR 定位为**可解释性/结构约束机制**（提供通路级训练轨迹与单调约束），明确报告 ΔAUC≈0，不做性能提升声明；
  论文卖点=「可解释的动态图学习 + 通路先验结构」而非「性能提升」。
- **统计口径**：DeLong 多重校正后无显著类别、Stack 增量 CI 重叠 → 正文用点估计+CI 报告，避免显著性断言；
  主表保留 macro-AUC/acc/kappa + bootstrap CI，P 值仅作探索性报告并注明未校正的局限。
- **时间外推**：TS 增益方向保持但幅度变小、校准偏差显著 → Limitations 如实说明。

## 四、下一阶段（写作）
- 待用户确认后按 CSBJ 规范启动正文（Methods → Results → Intro → Discussion，SCI 地道英文）
- 数字主库 `work/numbers_master.json` 已含全部核心数字；写作时统一 2 位小数、P 值规则（P≥0.001 三位、P<0.001 写 P<0.001）
- 图 Figure1A–3B（SVG 可编辑已就绪）、表 Table1–5、参考文献 52 条（DOI 已验证）
