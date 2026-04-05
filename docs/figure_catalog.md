# 图表目录与论文写作说明

本文档回答一个问题：当前仓库默认导出的每一张图，在论文里应该如何解释、如何使用、如何写图注。

使用约定：

- `README.md` 只负责索引
- `docs/visualization_guide.md` 只负责说明怎么调图
- 本文档只负责说明每张图表达什么、适合放在哪、正文里怎么写

---

## 1. 条目模板

每个条目统一包含：

- 状态
- 输出文件
- 生成入口
- 视觉元素
- 论文用途
- 推荐章节
- 推荐图注模板
- 推荐正文写法
- 审稿风险
- 主图 / 附录建议

---

## 2. Benchmark 默认图包

### 2.1 `benchmark_metrics_grid.png`

- 状态：已实现
- 输出文件：`benchmark_outputs/.../figures/benchmark_metrics_grid.png`
- 生成入口：`PerformanceComparisonPlots.plot_three_metrics_grid()`
- 视觉元素：列为问题，行为 `MIGD / SP / MS`，柱高为均值，最佳算法高亮
- 论文用途：benchmark 主结果总图
- 推荐章节：主实验结果
- 推荐图注模板：图 X 展示了各算法在不同动态多目标测试问题上的三项核心指标对比，其中更低的 MIGD/SP 和更高的 MS 表示更优的前沿质量。
- 推荐正文写法：如图 X 所示，KEMM 在多数问题上的逼近与分布质量更优，说明其在环境变化后具有更稳定的恢复能力。
- 审稿风险：若不配合统计图，容易被质疑缺少显著性支持
- 建议：主图

### 2.2 `benchmark_rank_bar.png`

- 状态：已实现
- 输出文件：`benchmark_outputs/.../figures/benchmark_rank_bar.png`
- 生成入口：`PerformanceComparisonPlots.plot_rank_bar()`
- 视觉元素：横轴为平均排名，纵轴为算法，越低越好
- 论文用途：总结综合排名
- 推荐章节：主实验结果收口
- 推荐图注模板：图 X 给出了所有问题与指标上的综合平均排名结果，较低排名表示更优的总体表现。
- 推荐正文写法：综合排名结果表明，KEMM 的收益并非来自单一问题，而是在整体 benchmark 上保持领先。
- 审稿风险：平均排名会压缩问题细节，需和逐问题结果一起读
- 建议：主图

### 2.3 `benchmark_heatmap.png`

- 状态：已实现
- 输出文件：`benchmark_outputs/.../figures/benchmark_heatmap.png`
- 生成入口：`PerformanceComparisonPlots.plot_heatmap_normalized()`
- 视觉元素：横轴为问题，纵轴为算法，颜色为归一化 MIGD，格内文本为原始值
- 论文用途：展示不同问题上性能结构差异
- 推荐章节：主实验结果 / 补充可视化
- 推荐图注模板：图 X 给出了各算法在不同测试问题上的归一化 MIGD 热图，其中颜色越深表示相对性能越优。
- 推荐正文写法：热图说明 KEMM 的优势并不局限于某一类问题，而是在多类动态场景下都具有竞争力。
- 审稿风险：热图只能表达概览，不能替代统计检验
- 建议：主图

### 2.4 `benchmark_wilcoxon.png`

- 状态：已实现
- 输出文件：`benchmark_outputs/.../figures/benchmark_wilcoxon.png`
- 生成入口：`StatisticalAnalysisPlots.plot_wilcoxon_heatmap()`
- 视觉元素：横轴为问题，纵轴为对比算法，颜色与单元格数字为 p 值
- 论文用途：展示统计显著性
- 推荐章节：统计分析
- 推荐图注模板：图 X 展示了基于 Wilcoxon 秩和检验的两两显著性比较结果，较小 p 值表示提出算法相对基线的统计优势更显著。
- 推荐正文写法：统计检验结果表明，KEMM 的优势不是偶然的单次运行现象，而在多个问题上具有统计支撑。
- 审稿风险：需要明确样本量与显著性阈值
- 建议：主图

### 2.5 `benchmark_pairwise_wins.png`

- 状态：已实现
- 输出文件：`benchmark_outputs/.../figures/benchmark_pairwise_wins.png`
- 生成入口：`StatisticalAnalysisPlots.plot_pairwise_win_matrix()`
- 视觉元素：矩阵单元表示行算法在 MIGD 上胜过列算法的问题数
- 论文用途：展示逐对比较关系
- 推荐章节：统计分析 / 附录
- 推荐图注模板：图 X 给出了算法间的 pairwise win matrix，用于展示 KEMM 与各基线逐对比较时的优势范围。
- 推荐正文写法：胜率矩阵表明，KEMM 对多数基线保持更高的逐问题胜出次数。
- 审稿风险：胜率不等同于显著性，需与 Wilcoxon 图联合使用
- 建议：主图或强附录图

### 2.6 `benchmark_igd_time.png`

- 状态：已实现
- 输出文件：`benchmark_outputs/.../figures/benchmark_igd_time.png`
- 生成入口：`ProcessAnalysisPlots.plot_igd_convergence()`
- 视觉元素：横轴为变化步，纵轴为 IGD，实线为均值，阴影为波动范围
- 论文用途：展示逼近恢复速度与稳定性
- 推荐章节：机制分析 / 收敛分析
- 推荐图注模板：图 X 展示了环境变化过程中的 IGD 收敛曲线，阴影区域表示多次重复运行下的波动范围。
- 推荐正文写法：KEMM 的 IGD 曲线下降更快且波动更小，说明其在变化后具有更快的恢复与更稳定的前沿质量。
- 审稿风险：需说明阴影代表标准差还是置信区间
- 建议：主图

### 2.7 `benchmark_hv_time.png`

- 状态：已实现
- 输出文件：`benchmark_outputs/.../figures/benchmark_hv_time.png`
- 生成入口：`ProcessAnalysisPlots.plot_hv_convergence()`
- 视觉元素：横轴为变化步，纵轴为 HV，实线为均值，阴影为波动范围
- 论文用途：补充展示覆盖能力恢复
- 推荐章节：机制分析 / 收敛分析
- 推荐图注模板：图 X 给出了变化过程中的超体积收敛曲线，用于衡量各算法对真实 Pareto 区域的覆盖恢复能力。
- 推荐正文写法：与 IGD 曲线结合可见，KEMM 不仅逼近更快，也能更快恢复对目标空间的覆盖。
- 审稿风险：需说明参考点设置方式与 HV 仅对 2 目标问题直接使用
- 建议：主图

### 2.8 `benchmark_operator_ratios.png`

- 状态：已实现
- 输出文件：`benchmark_outputs/.../figures/benchmark_operator_ratios.png`
- 生成入口：`ProcessAnalysisPlots.plot_operator_ratio_history()`
- 视觉元素：各曲线分别表示 `memory / prediction / transfer / reinit` 比例
- 论文用途：展示 KEMM 的自适应分配机制
- 推荐章节：机制分析
- 推荐图注模板：图 X 展示了 KEMM 在动态变化过程中的 operator 分配比例演化。
- 推荐正文写法：在剧烈变化区间，预测与重启比例提高，而在平稳阶段，记忆与迁移重新占主导。
- 审稿风险：单独展示内部状态说服力不足，需和性能图配合
- 建议：主图

### 2.9 `benchmark_response_quality.png`

- 状态：已实现
- 输出文件：`benchmark_outputs/.../figures/benchmark_response_quality.png`
- 生成入口：`ProcessAnalysisPlots.plot_response_quality_history()`
- 视觉元素：横轴为变化步，纵轴为 response quality proxy
- 论文用途：解释变化响应本身是否有效
- 推荐章节：机制分析
- 推荐图注模板：图 X 展示了 KEMM 在连续环境变化中的响应质量代理量变化。
- 推荐正文写法：响应质量在变化后逐步回升，说明融合候选池能较快恢复有效搜索。
- 审稿风险：需要给出 response quality 的定义来源
- 建议：主图

### 2.10 `benchmark_prediction_confidence.png`

- 状态：已实现
- 输出文件：`benchmark_outputs/.../figures/benchmark_prediction_confidence.png`
- 生成入口：`ProcessAnalysisPlots.plot_prediction_confidence()`
- 视觉元素：横轴为变化步，纵轴为 prediction confidence
- 论文用途：展示预测分支可信度何时上升或下降
- 推荐章节：机制分析
- 推荐图注模板：图 X 给出了漂移预测模块的置信度变化，用于刻画预测分支在不同环境阶段的可靠性。
- 推荐正文写法：预测置信度在相邻环境相似时更高，而在剧烈变化时明显下降，这与 operator 分配变化一致。
- 审稿风险：需要明确置信度来自模型输出还是后验估计
- 建议：主图

### 2.11 `benchmark_change_dashboard.png`

- 状态：已实现
- 输出文件：`benchmark_outputs/.../figures/benchmark_change_dashboard.png`
- 生成入口：`AlgorithmMechanismPlots.plot_change_diagnostics_dashboard()`
- 视觉元素：同页汇总 operator ratio、response quality、prediction confidence、change magnitude
- 论文用途：做机制总览
- 推荐章节：机制分析 / 可解释性展示
- 推荐图注模板：图 X 汇总展示了 KEMM 在动态变化过程中的多种机制诊断量，用于解释其变化响应策略。
- 推荐正文写法：dashboard 将策略分配与预测可靠性放在同一页，便于解释性能优势与内部机制间的对应关系。
- 审稿风险：信息密度高，正文需要点名解释关键子图
- 建议：主图或补充主图

### 2.12 `benchmark_ablation.png`

- 状态：已实现
- 输出文件：`benchmark_outputs/.../figures/benchmark_ablation.png`
- 生成入口：`AblationStudyPlots.plot_ablation_comparison()`
- 视觉元素：横轴为变体，纵轴为聚合 MIGD
- 论文用途：说明核心模块必要性
- 推荐章节：消融实验
- 推荐图注模板：图 X 比较了完整 KEMM 与关键消融变体的聚合性能差异，用于分析各模块的贡献。
- 推荐正文写法：完整 KEMM 相对简化变体仍保持优势，说明记忆、预测与迁移并非冗余组件。
- 审稿风险：需明确每个消融变体具体关闭了什么能力
- 建议：主图

---

## 3. Benchmark 附录 / 兼容图

### 3.1 `benchmark_migd_bar.png`

- 状态：已实现
- 输出文件：`benchmark_outputs/.../figures/benchmark_migd_bar.png`
- 生成入口：`PerformanceComparisonPlots.plot_migd_main_table()`
- 视觉元素：横轴为问题，纵轴为各算法的 MIGD 均值，颜色区分算法
- 论文用途：传统单指标对比图
- 推荐章节：附录 / 兼容结果对齐
- 推荐图注模板：图 X 给出了各算法在不同 benchmark 问题上的 MIGD 条形对比，其中柱高越低表示与真实 Pareto 前沿越接近。
- 推荐正文写法：该图适合与旧版本实验对齐，但由于只展示单指标，正文结论应以多指标主图和统计图为准。
- 审稿风险：与 `benchmark_metrics_grid.png` 高度重复
- 建议：附录，仅在 `appendix_plots=True` 时导出

---

## 4. Ship 默认图包

### 4.1 `*_environment_overlay.png`

- 状态：已实现
- 输出文件：`ship_simulation/outputs/.../figures/<scenario>_environment_overlay.png`
- 生成入口：`save_environment_overlay()`
- 视觉元素：热力背景为环境标量场，箭头为矢量场，阴影为障碍，彩色轨迹为代表执行轨迹，虚线为目标船轨迹
- 论文用途：交代场景、环境和代表轨迹
- 推荐章节：案例分析 / 物理仿真结果
- 推荐图注模板：图 X 给出了近海会遇场景下的环境场叠加轨迹图，其中背景热图表示环境暴露或风险势场，箭头表示流场方向与强度，阴影区域表示静态障碍或禁航区。
- 推荐正文写法：环境叠加图表明，KEMM 生成的轨迹在避开高代价区域的同时保持了合理的整体推进效率。
- 审稿风险：需要明确热图具体对应的物理量
- 建议：主图

### 4.2 `*_snapshots.png`

- 状态：已实现
- 输出文件：`ship_simulation/outputs/.../figures/<scenario>_snapshots.png`
- 生成入口：`save_dynamic_avoidance_snapshots()`
- 视觉元素：多子图时间切片、本船位置、目标船位置、相对距离连线
- 论文用途：展示关键避碰瞬间
- 推荐章节：案例分析 / 动态避碰分析
- 推荐图注模板：图 X 展示了动态避碰过程中的时间切片快照，各子图对应不同时间点的本船与目标船相对位置。
- 推荐正文写法：快照结果表明，本船在关键交会时刻提前调整航迹，从而保持正的安全净空。
- 审稿风险：若缺少时间标注，会削弱“动态”表达
- 建议：主图

### 4.3 `*_spatiotemporal.png`

- 状态：已实现
- 输出文件：`ship_simulation/outputs/.../figures/<scenario>_spatiotemporal.png`
- 生成入口：`save_spatiotemporal_plot()`
- 视觉元素：X/Y 为空间，Z 为时间，曲线表示船舶时空轨迹
- 论文用途：展示动态交通环境中的时空穿越关系
- 推荐章节：案例分析
- 推荐图注模板：图 X 给出了场景的三维时空规划结果，其中 Z 轴表示时间，轨迹曲线展示了本船和目标船的时空关系。
- 推荐正文写法：三维时空视角强调了本船并非只在空间上绕开目标船，而是在时间轴上错峰通过危险区域。
- 审稿风险：3D 图需配合 2D 图一起读
- 建议：主图

### 4.4 `*_control_timeseries.png`

- 状态：已实现
- 输出文件：`ship_simulation/outputs/.../figures/<scenario>_control_timeseries.png`
- 生成入口：`save_control_time_series()`
- 视觉元素：航向、等效转向命令、实际转向角速度、速度四个子图
- 论文用途：验证轨迹可执行性
- 推荐章节：动力学约束分析
- 推荐图注模板：图 X 展示了代表轨迹的航向、控制输入、转向角速度与速度时序，用于验证规划结果未违反简化船模约束。
- 推荐正文写法：这些时序量说明轨迹不是几何上好看但无法执行的路径，而是与动力学限制一致的控制过程。
- 审稿风险：要说明控制量是等效命令而非真实舵角
- 建议：主图

### 4.5 `*_pareto3d.png`

- 状态：已实现
- 输出文件：`ship_simulation/outputs/.../figures/<scenario>_pareto3d.png`
- 生成入口：`save_pareto_3d_with_knee()`
- 视觉元素：三维目标空间散点、knee point 高亮、Fuel-Time 局部放大 inset
- 论文用途：展示三目标折中结构与推荐解
- 推荐章节：主结果 / 决策分析
- 推荐图注模板：图 X 展示了三目标空间中的 Pareto 前沿，并以星标高亮 knee point，同时给出局部放大视图以展示其邻域解分布。
- 推荐正文写法：knee point 位于能耗、时间与风险的折中区域，局部放大图进一步说明该点并非孤立异常值，而是前沿拐点附近的稳定选择。
- 审稿风险：需说明 knee 检测准则与局部放大的投影视角
- 建议：主图

### 4.6 `*_parallel.png`

- 状态：已实现
- 输出文件：`ship_simulation/outputs/.../figures/<scenario>_parallel.png`
- 生成入口：`save_parallel_coordinates()`
- 视觉元素：横轴为多项归一化指标，折线为各算法代表轨迹，当前统一为“越高越好”方向
- 论文用途：展示多分析指标对比
- 推荐章节：补充多指标分析
- 推荐图注模板：图 X 使用 parallel coordinates 对比不同算法在多项归一化指标上的综合表现。
- 推荐正文写法：在 3 个主目标之外，parallel coordinates 进一步揭示了净空、平滑性和控制代价的差异。
- 审稿风险：必须明确归一化方向和尺度
- 建议：主图或补充主图

### 4.7 `*_radar.png`

- 状态：已实现
- 输出文件：`ship_simulation/outputs/.../figures/<scenario>_radar.png`
- 生成入口：`save_radar_chart()`
- 视觉元素：多指标雷达图，当前统一为“越高越好”方向
- 论文用途：直观比较多维综合表现
- 推荐章节：案例对比 / 补充图
- 推荐图注模板：图 X 以雷达图形式比较了不同算法代表轨迹在能耗、时间、风险与控制代价等指标上的归一化表现。
- 推荐正文写法：雷达图从直观上展示了 KEMM 在安全性和执行性指标上的综合优势。
- 审稿风险：雷达图面积不能当成严格统计量
- 建议：补充主图

### 4.8 `*_convergence.png`

- 状态：已实现
- 输出文件：`ship_simulation/outputs/.../figures/<scenario>_convergence.png`
- 生成入口：`save_convergence_statistics()`
- 视觉元素：均值收敛曲线与阴影带
- 论文用途：展示 ship 场景上的优化稳定性
- 推荐章节：算法分析 / 重复运行统计
- 推荐图注模板：图 X 展示了多次重复运行下各算法在 ship 场景中的加权目标收敛曲线，阴影表示跨运行波动范围。
- 推荐正文写法：收敛曲线显示 KEMM 在早期搜索阶段已具备较好的恢复速度，同时跨运行波动较小。
- 审稿风险：需解释 surrogate 分数的组成方式
- 建议：主图

### 4.9 `*_distribution.png`

- 状态：已实现
- 输出文件：`ship_simulation/outputs/.../figures/<scenario>_distribution.png`
- 生成入口：`save_distribution_violin()`
- 视觉元素：Fuel / Time / Risk 的 violin 分布图
- 论文用途：比较重复运行分布
- 推荐章节：统计对比
- 推荐图注模板：图 X 展示了不同算法在多次重复运行下的 Fuel、Time 与 Risk 分布，用于分析稳定性与整体表现。
- 推荐正文写法：分布图表明，KEMM 的优势不仅体现在单次最优值，也体现在整体运行分布更集中。
- 审稿风险：运行次数过少时解释力度不足
- 建议：主图或强附录图

### 4.10 `*_dashboard.png`

- 状态：已实现
- 输出文件：`ship_simulation/outputs/.../figures/<scenario>_dashboard.png`
- 生成入口：`save_summary_dashboard()`
- 视觉元素：环境叠加、雷达图、收敛子图和文字摘要的一页式组合
- 论文用途：做总览页，不替代拆分主图
- 推荐章节：补充材料 / 答辩总览页 / 仓库展示
- 推荐图注模板：图 X 给出了场景结果的一页式总览，其中综合展示了代表轨迹、指标对比和收敛摘要。
- 推荐正文写法：dashboard 适合用于快速概览，但正文分析仍应回到拆分主图逐项展开。
- 审稿风险：信息密度高，不适合作为唯一结果图
- 建议：补充图

---

## 5. Ship 附录 / 兼容图

### 5.1 `appendix_objective_bars.png`

- 状态：已实现
- 输出文件：`ship_simulation/outputs/.../figures/appendix_objective_bars.png`
- 生成入口：`save_normalized_objective_bars()`
- 视觉元素：横轴为场景，纵轴为归一化目标值，柱色区分 KEMM 与随机基线
- 论文用途：传统条形对比图
- 推荐章节：附录 / 兼容结果对齐
- 推荐图注模板：图 X 展示了不同场景下代表解在归一化目标空间中的柱形对比，用于快速查看 KEMM 与基线在主目标上的差异。
- 推荐正文写法：该图适合用于补充说明主目标差异，但其信息密度和表达力弱于 Pareto、parallel 与 dashboard 主图。
- 审稿风险：归一化方式若未说明，容易被误读为绝对量比较
- 建议：附录，仅在 `appendix_plots=True` 时导出

### 5.2 `appendix_risk_bars.png`

- 状态：已实现
- 输出文件：`ship_simulation/outputs/.../figures/appendix_risk_bars.png`
- 生成入口：`save_risk_bars()`
- 视觉元素：横轴为场景，纵轴为风险相关统计量，柱色区分算法
- 论文用途：传统风险条形对比图
- 推荐章节：附录 / 风险结果补充
- 推荐图注模板：图 X 给出了不同场景下风险相关统计量的条形对比，用于补充说明 KEMM 与基线在安全性上的差异。
- 推荐正文写法：风险条形图适合作为补充证据，但不能替代时间历程、快照或净空指标对安全性的动态解释。
- 审稿风险：静态汇总柱图会掩盖风险的时间结构
- 建议：附录，仅在 `appendix_plots=True` 时导出

### 5.3 `*_risk_series.png` / `*_speed_profile.png` / `*_trajectory.png`

- 状态：兼容保留
- 输出文件：兼容层函数按旧命名调用时生成
- 生成入口：`save_risk_time_series()` / `save_speed_profiles()` / `save_trajectory_comparison()`
- 视觉元素：分别对应风险时间历程、速度时间历程和二维轨迹叠加
- 论文用途：结果对齐或调试
- 推荐章节：附录 / 调试补充
- 推荐图注模板：图 X 给出了旧版兼容图输出，用于与历史版本结果或调试记录保持一致。
- 推荐正文写法：这些图可用于补充解释单一过程量，但当前正文应优先使用新的论文级核心图包。
- 审稿风险：若单独引用旧兼容图，容易与新版图包叙事重复或冲突
- 建议：附录，不作为默认主图

---

## 6. 已实现的近期增强图

### 6.1 `*_route_planning_panel.png`

- 状态：已实现
- 输出文件：`ship_simulation/outputs/.../figures/<scenario>_route_planning_panel.png`
- 生成入口：`save_route_planning_panel()`
- 视觉元素：二维海域主图、密障碍背景、目标船速度箭头、瓶颈局部 inset
- 论文用途：展示复杂受限海域中的路线规划主结果
- 推荐正文写法：该图适合放在高密障碍场景分析开头，用来说明算法如何在有限通道中兼顾推进与安全。
- 建议：主图

### 6.2 `*_pareto_projection.png`

- 状态：已实现
- 输出文件：`ship_simulation/outputs/.../figures/<scenario>_pareto_projection.png`
- 生成入口：`save_pareto_projection_panel()`
- 视觉元素：Fuel-Time、Fuel-Risk、Time-Risk 三个二维投影散点及 projected frontier curve
- 论文用途：补强 3D Pareto 图的可读性
- 推荐正文写法：二维投影揭示了三目标前沿在成对目标上的折中结构，便于解释 knee 点邻域的局部几何。
- 建议：主图

### 6.3 `*_risk_breakdown.png`

- 状态：已实现
- 输出文件：`ship_simulation/outputs/.../figures/<scenario>_risk_breakdown.png`
- 生成入口：`save_risk_breakdown_time_series()`
- 视觉元素：总风险曲线、阈值线与 `domain / dcpa / obstacle / environment` 分解项
- 论文用途：解释风险是如何随时间累积和转移的
- 推荐正文写法：该图适合说明关键风险峰值到底来自会遇几何、障碍净空还是环境暴露，而不是只给单个总风险数值。
- 建议：主图

### 6.4 `*_safety_envelope.png`

- 状态：已实现
- 输出文件：`ship_simulation/outputs/.../figures/<scenario>_safety_envelope.png`
- 生成入口：`save_safety_envelope_plot()`
- 视觉元素：总体净空、静态障碍净空、船间距离，以及 DCPA/TCPA 与 COLREG scale
- 论文用途：把“避障”和“避船”两个安全维度拆开解释
- 推荐正文写法：安全包络图特别适合说明拥挤海域中哪一段是障碍瓶颈，哪一段是会遇瓶颈。
- 建议：主图

### 6.5 `*_run_statistics.png`

- 状态：已实现
- 输出文件：`ship_simulation/outputs/.../figures/<scenario>_run_statistics.png`
- 生成入口：`save_run_statistics_panel()`
- 视觉元素：success rate、minimum clearance、minimum ship distance、runtime 的 grouped bars + error bars
- 论文用途：补充 repeated-run 层面的安全与效率统计
- 推荐正文写法：该图可与 violin 和 convergence 联用，说明提出方法不仅单次轨迹更好，而且跨运行更稳。
- 建议：主图或强附录图

---

## 7. 维护约定

只要默认导出图表发生变化，必须同步更新：

- `README.md`
- `docs/visualization_guide.md`
- `docs/figure_catalog.md`
- `docs/ship_simulation_reference.md`
- `docs/kemm_reference.md`

新增图表条目时，至少补以下内容：

1. 输出文件名模式
2. 生成函数
3. 图意与视觉编码说明
4. 推荐图注模板
5. 主图还是附录图的建议
