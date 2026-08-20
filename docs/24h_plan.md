# 24 小时任务计划（2026-08-16 更新）

> 状态：**CSBJ 正文初稿已完成（01–05 五章英文手稿）**；58 条真实参考文献全部插入正文并按首次引用编号（DOI 经 CrossRef/DataCite 双登记库逐条验证）。
> 关键决策：TPR 通路先验定位为可解释性/结构约束机制（预测贡献≈0，已三方验证），不做性能提升声明。

## 已完成：补强 8 项（2026-08-14/15）
- [x] λ 网格扫描 + bootstrap ΔAUC CI：`work/lambda_sweep_final_report.md`、`results/lambda_sweep_results_bootstrap.csv`
- [x] 负对照 anti/perm：`work/negctrl_report.md`、`results/dggru_tpr_anti_oof.npz`、`results/dggru_tpr_perm_oof.npz`
- [x] 时间验证 + 时间窗消融：`work/temporal_validation_report.md`、`results/temporal_validation_results.csv`
- [x] 统计校正 + 亚组：`work/stats_fixes.md`、`results/stats_fixes_results.csv`、`results/subgroup_results.csv`
- [x] 通路敏感性：`work/pathway_sensitivity.md`、`results/pathway_sensitivity_results.csv`
- [x] 标签鲁棒：`work/label_robustness.md`、`results/label_robustness_results.csv`
- [x] 种子稳健：`work/seed_robustness.md`、`results/seed_robustness_results.csv`
- [x] 泄漏审计：`work/leak_audit_report.md`
- [x] 汇总：`docs/strengthening_report.md`

## 已完成：正文初稿（2026-08-16）
- [x] 五章手稿：`manuscript/01_front_matter.md`（标题页+摘要+引言）、`02_methods.md`、`03_results.md`（含图注表题）、`04_discussion.md`、`05_declarations.md`
- [x] 全文引用编号修复：`[key]` 碎片全部映射为数字编号；正文首次出现顺序严格单调 1–58
- [x] 正式参考文献列表：`manuscript/05_declarations.md` 第 7 节 58 条；`docs/references.md`、`docs/references.bib` 同步重建
- [x] DOI 验证：52 条 CrossRef + 9 条 DataCite（10.48550 arXiv），合计 58/58 VERIFIED；修复 ref50 EvolveGCN DOI 错配
- [x] 新增 6 条真实文献：GEO 四队列出处（GSE37587/58294/16561/22255）、TRIPOD+AI 2024、出院去向 ML 文献
- [x] 数字一致性核查：摘要/结果/讨论 acc 0.54、kappa 0.41、宏 AUC 0.830、CI +0.0005~+0.0084、负对照 0.8146/0.8152/0.8147/0.8136、λ 网格 0.8140–0.8172 一致
- [x] P 值规则统一：P≥0.001 保留 3 位小数；P<0.001 一律写 P < 0.001
- [x] 图表编号核查：Figure 1A–1D / 2A–2B / 3A–3B、Table 1–5 按出现顺序命名

## 待办（需要用户提供信息或下一步处理）
- [ ] 作者/单位/通讯作者、基金号、致谢、CRediT 分工：`manuscript/01_front_matter.md`、`05_declarations.md` 中 [to be completed] 占位
- [ ] 代码仓库 URL 与归档 DOI：`05_declarations.md` Code availability 占位
- [ ] 图表 SVG/PNG 终稿（Figure 1–3，可编辑文字 SVG + 300dpi PNG）
- [ ] 投稿材料：Cover letter、Highlights、Graphical abstract 草图
- [ ] 终检：CSBJ 格式限制（字数/图件数/参考文献格式）、全文 SCI 地道英文润色、AI 使用声明确认

## 新增：投稿竞争力补强 Batch 2（2026-08-17）
- [x] 数值口径统一（折均值）：正文 3.4 / Table 2 / Table 3 / numbers_master 一致；词数 5762；摘要补窄口径灵敏度（287 词）
- [x] 风格审计修复：em dash 0、术语统一、P 值格式 0 违规；Discussion 4.5 新增第八条限制（深模型未直接外验，如实声明）
- [ ] 深模型 harmonized 重训 + eICU 直接迁移外验（Noether 代理运行中）
- [ ] 公开基线公平对比 GRU-D / Transformer / STGCN（Kant 代理运行中）
- [ ] 代码仓库 README + LICENSE + 复现说明（本地起草中）
- [ ] 备选：AmsterdamUMCdb 三源外验（需下载，PhysioNet 认证已具备）
- [ ] mRS/90 天随访结局验证：需外部数据集或临床合作，投稿前大概率不可行，作为 limitation

- [x] MIMIC-III 外验方案已定稿（`docs/mimic3_extval_plan.md`）：ICD-9 430-438、2001-2007 年代不相交主分析、6 级序数结局、64x8 序列 + 173 维静态特征复用 harmonized schema
- [ ] MIMIC-III 数据下载（需用户 PhysioNet 接受 DUA，约 6.7 GB，放置于 mimic-iii-1.4/）
- [ ] MIMIC-III harmonized 构建脚本（`src/data/build_mimic3_harmonized.py`，数据到位后立即执行）

- [x] MIMIC-III v1.4 已下载完成（6.17GB，mimic-iii-clinical-database-1.4_1/）
- [x] 深模型 eICU 迁移外验代理（Noether）、基线对比代理（Kant）、MIMIC-III 构建+外验代理（Huygens）三路并行运行中
