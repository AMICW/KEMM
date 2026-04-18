# 运行命令手册

这份文档是仓库的中文运行速查表，面向已经装好环境、只想快速找到当前真实入口和常用参数的人。

如果你还没装环境，先看 [environment_setup.md](environment_setup.md)。
如果你需要英文版命令说明，转到 [how_to_run.md](how_to_run.md)。
如果你想看整个文档地图，转到 [README.md](README.md)。

## 1. 真实入口

benchmark 真实入口：

- `python -m apps.benchmark_runner`

ship 批量报告真实入口：

- `python ship_simulation/run_report.py`

兼容层入口仍然可用，但不再是首选：

- `python run_experiments.py`
- `python -m apps.ship_runner`

---

## 2. Benchmark

### 2.1 快速 smoke test

```powershell
python -m apps.benchmark_runner --quick --plot-preset paper
```

当前 `--quick` 的含义：

- 重复运行：`2`
- 环境变化次数：`5`
- 每次变化后的进化代数：`10`
- 问题集合：`FDA1`, `FDA3`, `dMOP2`
- 默认 worker：`1`

### 2.2 完整 benchmark

README 只保留这一条完整入口：

```powershell
python -m apps.benchmark_runner --full --workers 4
```

当前 `--full` 的含义：

- 重复运行：`5`
- 环境变化次数：`10`
- 每次变化后的进化代数：`20`
- 问题集合：`FDA1`, `FDA2`, `FDA3`, `dMOP1`, `dMOP2`, `dMOP3`
- 动态设置 sweep：`(5,10)`, `(10,10)`, `(10,20)`
- 默认包含消融与全部主图
- 默认启用 benchmark 任务级缓存

### 2.3 带 JY 问题的完整 benchmark

```powershell
python -m apps.benchmark_runner --with-jy --workers 4
```

这会在标准问题集基础上再加入 `JY1`, `JY4`。

### 2.4 benchmark 缓存与重跑

benchmark 现在默认对完整运行启用任务级缓存，缓存粒度固定为：

- `(setting, algorithm, problem, run, ablation_variant)`

缓存目录：

- `benchmark_outputs/_cache/benchmark_tasks/`

同配置重跑时，默认会直接复用缓存结果。

强制忽略缓存：

```powershell
python -m apps.benchmark_runner --full --force-rerun
```

做一个明显的 warm rerun 对比：

```powershell
python -m apps.benchmark_runner --quick --force-rerun
python -m apps.benchmark_runner --quick
```

### 2.5 常用附加参数

跳过消融：

```powershell
python -m apps.benchmark_runner --full --skip-ablation
```

只导出原始表格和 Markdown summary，不渲染图：

```powershell
python -m apps.benchmark_runner --full --summary-only
```

指定算法和问题子集：

```powershell
python -m apps.benchmark_runner --quick --algorithms KEMM RI Tr
python -m apps.benchmark_runner --quick --problems FDA1 FDA3 dMOP2
```

自定义输出目录：

```powershell
python -m apps.benchmark_runner --full --output-dir benchmark_outputs\my_run
```

切换图表 preset 和 SciencePlots 风格：

```powershell
python -m apps.benchmark_runner --full --plot-preset ieee
python -m apps.benchmark_runner --full --science-style science,ieee,no-latex
```

导出附录图和交互 figure bundle：

```powershell
python -m apps.benchmark_runner --full --appendix-plots --interactive-figures
```

### 2.6 benchmark 输出里优先看什么

完整运行后，优先看这些文件：

- `reports/summary.md`
- `raw/summary.json`
- `raw/paper_table_metrics.csv`
- `raw/ablation_delta_metrics.csv`
- `figures/benchmark_migd_table.png`
- `figures/benchmark_ablation.png`

补充说明：

- `benchmark_migd_table.png` 是论文风格主表，对应三组动态设置。
- `benchmark_ablation.png` 画的是相对 `KEMM-Full` 的 `MIGD` 退化百分比，正值表示删掉模块后更差。
- 只有 canonical setting 会完整保留 `igd_curve / hv_curve / change_diagnostics` 聚合；非 canonical setting 主要保留最终标量指标。

### 2.7 兼容层入口

```powershell
python run_experiments.py --quick
python run_experiments.py --full
```

这些命令仍可用，但真实入口仍然是：

```powershell
python -m apps.benchmark_runner
```

---

## 3. Ship 物理仿真

### 3.1 先区分两个入口

单次 demo：

```powershell
python -m apps.ship_runner
```

完整批量报告：

```powershell
python ship_simulation/run_report.py
```

如果你要跑完整 ship 实验模块，用第二条。

### 3.2 单次 demo

```powershell
python -m apps.ship_runner
python -c "from ship_simulation.main_demo import run_demo; run_demo('crossing', optimizer_name='kemm', show_animation=False)"
python -c "from ship_simulation.main_demo import run_demo; run_demo('harbor_clutter', optimizer_name='kemm', show_animation=False)"
```

适合：

- 单场景调试
- 只看一条轨迹
- 不生成完整报告

### 3.3 快速 ship 报告

```powershell
python ship_simulation/run_report.py --quick --scenarios crossing --n-runs 1 --plot-preset paper
python ship_simulation/run_report.py --quick --scenarios crossing harbor_clutter --n-runs 1 --plot-preset paper
```

当前 `--quick` 的含义：

- `scenario_profiles.active_profile_name = legacy_uniform`
- `random_search_samples = 20`
- `NSGA-style pop_size = 22`
- `NSGA-style generations = 10`
- `KEMM pop_size = 28`
- `KEMM generations = 12`
- `KEMM initial_guess_copies = 4`
- `local_horizon = 320`
- `execution_horizon = 160`
- `max_replans = 8`
- `n_runs = 1`
- `render_workers = 1`

### 3.4 完整 ship 报告

README 只保留这一条完整入口：

```powershell
python ship_simulation/run_report.py --workers 4
```

默认会跑：

- 场景：`head_on`, `crossing`, `overtaking`, `harbor_clutter`
- 算法：`kemm`, `nsga_style`, `random`
- 每算法每场景：`3` 次
- 总 episode 数：`36`
- 默认图数：`4 * 19 + 2 = 78`

完整模式默认启用两件事：

- `scenario_profiles.active_profile_name = full_tuned`
- `episode_cache_enabled = True`

同时内部执行已经改成三阶段：

1. 先把 36 个 episode 全部算完并写入 `raw/episode_cache/`
2. 再并行渲染图表
3. 最后写 summary / metadata / inventory

### 3.5 ship 缓存与 metadata

ship episode 缓存目录：

- `ship_simulation/outputs/report_YYYYMMDD_HHMMSS/raw/episode_cache/`

同一输出目录、同一配置再次运行时：

- 会直接复用 episode cache
- 会完整重生成图表
- 会重写 summary / metadata / inventory

metadata 里现在会写：

- `scenario_solve_profile`
- `episode_compute_seconds`
- `figure_render_seconds`
- `episode_cache_hits`
- `episode_cache_misses`

### 3.6 选择场景、算法和重复次数

只跑高密港区：

```powershell
python ship_simulation/run_report.py --scenarios harbor_clutter --n-runs 3 --plot-preset paper
```

只比较 `kemm` 和 `random`：

```powershell
python ship_simulation/run_report.py --algorithms kemm random --scenarios crossing overtaking
```

追加“同预算、同重规划频率”的严格可比基线：

```powershell
python ship_simulation/run_report.py --algorithms kemm random nsga_style --strict-comparable
```

完整场景列表显式写出：

```powershell
python ship_simulation/run_report.py --scenarios head_on crossing overtaking harbor_clutter --n-runs 3 --plot-preset paper
```

### 3.7 动态 experiment profile

`--experiment-profile` 控制的是滚动重规划过程中的计划变化事件，不是场景预算 profile。

可选值：

- `baseline`
- `drift`
- `shock`
- `recurring_harbor`

命令示例：

```powershell
python ship_simulation/run_report.py --scenarios harbor_clutter --n-runs 3 --experiment-profile drift
python ship_simulation/run_report.py --scenarios harbor_clutter --n-runs 3 --experiment-profile shock
python ship_simulation/run_report.py --scenarios harbor_clutter --n-runs 3 --experiment-profile recurring_harbor
```

含义速记：

- `baseline`：不额外注入计划变化
- `drift`：逐步增强环境和交通漂移
- `shock`：突发 closure 和更强扰动
- `recurring_harbor`：港区漂移后部分回到熟悉模式

### 3.8 solve profile 的当前默认口径

ship 现在同时有第二套 profile 体系：`DemoConfig.scenario_profiles`。

当前口径：

- 完整模式默认：`full_tuned`
- quick 模式默认：`legacy_uniform`

`full_tuned` 会按场景分别调整：

- 求解预算
- 局部/执行时域
- `safety_clearance`
- 风险相关惩罚和 `domain_risk_weight`
- `objective_weights`

当前 CLI 没有单独暴露切换 solve profile 的参数；如果需要做 `legacy_uniform` 与 `full_tuned` 的 A/B，对代码入口做小改或在 Python 中调用 `build_default_demo_config()` 再改 `demo.scenario_profiles.active_profile_name`。

### 3.9 统计显著性导出

ship 报告现在默认导出置信区间和显著性检验结果：

- `raw/statistical_tests.json`
- `raw/statistical_tests.csv`
- `reports/statistical_significance.md`

检验规则：

- 样本量足够且通过正态性检查时，使用 Welch t-test
- 否则使用 Mann-Whitney U

### 3.10 鲁棒性扰动扫描

运行扰动强度扫描并导出成功率曲线：

```powershell
python ship_simulation/run_report.py --robustness-sweep --robustness-levels 0,0.25,0.5,0.75,1.0 --robustness-scenarios crossing overtaking harbor_clutter
```

输出文件：

- `raw/robustness_runs.csv`
- `raw/robustness_curve.csv`
- `raw/robustness_summary.json`
- `reports/robustness_sweep.md`
- `figures/robustness_success_curve.png`（启用渲染时）

### 3.11 图表与交互导出

ship 侧只有 `pareto3d` 和 `spatiotemporal` 会额外导出交互文件。

```powershell
python ship_simulation/run_report.py --quick --scenarios crossing --interactive-figures --interactive-html
```

说明：

- `*.fig.pickle`：matplotlib figure bundle
- `.html`：支持浏览器旋转的交互输出
- 其他 2D 图仍只导出 PNG

### 3.12 常用附加参数

只导出原始结果和 Markdown summary，不渲染图：

```powershell
python ship_simulation/run_report.py --summary-only
```

切换 preset 和 SciencePlots 风格：

```powershell
python ship_simulation/run_report.py --plot-preset ieee
python ship_simulation/run_report.py --science-style science,ieee,no-latex
```

导出附录图：

```powershell
python ship_simulation/run_report.py --appendix-plots
```

### 3.13 最新结构冒烟验证

如果你想用一条命令同时检查严格可比、统计导出和鲁棒性导出链路：

```powershell
python ship_simulation/run_report.py --quick --summary-only --scenarios crossing --n-runs 1 --algorithms kemm random --strict-comparable --robustness-sweep --robustness-levels 0,0.5 --robustness-scenarios crossing
```

预期关键输出：

- `raw/statistical_tests.json`
- `raw/statistical_tests.csv`
- `reports/statistical_significance.md`
- `raw/robustness_runs.csv`
- `raw/robustness_curve.csv`
- `raw/robustness_summary.json`
- `reports/robustness_sweep.md`

---

## 4. 输出目录

benchmark：

- `benchmark_outputs/benchmark_YYYYMMDD_HHMMSS/`

ship：

- `ship_simulation/outputs/report_YYYYMMDD_HHMMSS/`

两条主线共同的主子目录：

- `figures/`
- `raw/`
- `reports/`

ship 的 `raw/` 还特别值得看：

- `report_metadata.json`
- `figure_manifest.json`
- `representative_runs.json`
- `planning_steps.json`
- `scenario_catalog.json`

---

## 5. 测试与回归

完整测试：

```powershell
python -m unittest discover -s tests -v
```

benchmark 快速回归：

```powershell
python -m apps.benchmark_runner --quick --force-rerun
python -m apps.benchmark_runner --quick
```

ship 快速回归：

```powershell
python ship_simulation/run_report.py --quick --scenarios crossing harbor_clutter --n-runs 1
```

---

## 6. 建议阅读顺序

- [README.md](README.md)
- [how_to_run.md](how_to_run.md)
- [ship_experiment_playbook.md](ship_experiment_playbook.md)
- [visualization_guide.md](visualization_guide.md)
- [figure_catalog.md](figure_catalog.md)
