# 运行命令手册

这份文档是仓库的中文运行速查表。

适合下面两类场景：

- 你已经装好环境，只是想快速找到当前主线命令
- 你在 Windows / PowerShell 环境下频繁做 benchmark 和 ship 回归

如果你还没装环境，先看 [environment_setup.md](environment_setup.md)。
如果你需要英文版命令说明，转到 [how_to_run.md](how_to_run.md)。
如果你想看整个文档地图，转到 [README.md](README.md)。

## 1. 真实入口

benchmark 真实入口：

- `python -m apps.benchmark_runner`

ship 物理批量报告真实入口：

- `python ship_simulation/run_report.py`

如果你想运行当前主线，优先使用这两个入口。

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

### 2.2 标准完整 benchmark

```powershell
python -m apps.benchmark_runner --full --plot-preset paper
```

当前 `--full` 的含义：

- 重复运行：`5`
- 环境变化次数：`10`
- 每次变化后的进化代数：`20`
- 问题集合：`FDA1`, `FDA2`, `FDA3`, `dMOP1`, `dMOP2`, `dMOP3`
- 动态设置 sweep：`(5,10)`, `(10,10)`, `(10,20)`

### 2.3 带 JY 问题的完整 benchmark

```powershell
python -m apps.benchmark_runner --with-jy --plot-preset paper
```

这会在标准问题集基础上再加入 `JY1`, `JY4`。

### 2.4 中等规模 benchmark

当前 CLI 没有单独的 `--medium` 开关。

如果你想跑一组介于 `quick` 和 `full` 之间的实验，直接看：

- `docs/how_to_run.md`

里面提供了一段不改源码文件的 PowerShell 临时脚本，默认采用：

- `N_RUNS = 3`
- `N_CHANGES = 7`
- `GENS_PER_CHANGE = 15`

### 2.5 benchmark 输出里最重要的文件

完整运行后，优先看这些文件：

- `reports/summary.md`
- `raw/summary.json`
- `raw/paper_table_metrics.csv`
- `raw/ablation_delta_metrics.csv`
- `figures/benchmark_migd_table.png`
- `figures/benchmark_ablation.png`

解读要点：

- `benchmark_migd_table.png` 是论文风格主表，对应三组动态设置。
- `benchmark_ablation.png` 画的是相对 `KEMM-Full` 的 `MIGD` 退化百分比，所以正值表示删掉模块后更差。
- 如果你看到某份旧报告里 `FDA1` 和 `dMOP3` 指标整行一样，那是修复前的旧结果，不应继续引用。

### 2.6 常用附加参数

```powershell
python -m apps.benchmark_runner --full --output-dir benchmark_outputs\my_run
python -m apps.benchmark_runner --full --appendix-plots --interactive-figures
python -m apps.benchmark_runner --full --plot-preset ieee
```

### 2.7 兼容层入口

```powershell
python run_experiments.py --quick
python run_experiments.py --full
```

这些兼容命令仍可用，但真实入口仍是：

```powershell
python -m apps.benchmark_runner
```

## 3. Ship 物理仿真

### 3.1 先区分两个入口

单次 demo：

```powershell
python -m apps.ship_runner
```

完整物理批量报告：

```powershell
python ship_simulation/run_report.py
```

如果你要跑完整 ship 物理测试模块，用第二条。

### 3.2 单次 demo

```powershell
python -m apps.ship_runner
python -c "from ship_simulation.main_demo import run_demo; run_demo('crossing', optimizer_name='kemm', show_animation=False)"
python -c "from ship_simulation.main_demo import run_demo; run_demo('harbor_clutter', optimizer_name='kemm', show_animation=False)"
```

### 3.3 快速 ship 报告

```powershell
python ship_simulation/run_report.py --quick --scenarios crossing --n-runs 1 --plot-preset paper
python ship_simulation/run_report.py --quick --scenarios crossing harbor_clutter --n-runs 1 --plot-preset paper
```

当前 `--quick` 的含义：

- `random_search_samples = 20`
- `NSGA-style pop_size = 22`
- `NSGA-style generations = 10`
- `KEMM pop_size = 28`
- `KEMM generations = 12`
- `n_runs = 1`

### 3.4 完整 ship 物理测试

```powershell
python ship_simulation/run_report.py
python ship_simulation/run_report.py --plot-preset paper
python ship_simulation/run_report.py --scenarios head_on crossing overtaking harbor_clutter --n-runs 3 --plot-preset paper
```

默认会跑：

- 场景：`head_on`, `crossing`, `overtaking`, `harbor_clutter`
- 算法：`kemm`, `nsga_style`, `random`
- 每算法每场景：`3` 次

### 3.5 交互图导出

```powershell
python ship_simulation/run_report.py --quick --scenarios crossing --interactive-figures --interactive-html
```

说明：

- ship 侧只有 `pareto3d` 和 `spatiotemporal` 会额外导出 `*.fig.pickle` / `.html`
- 其他 2D 图只导出 PNG

## 4. 测试

完整测试：

```powershell
python -m unittest discover -s tests -v
```

## 5. 输出目录

benchmark：

- `benchmark_outputs/benchmark_YYYYMMDD_HHMMSS/`

ship：

- `ship_simulation/outputs/report_YYYYMMDD_HHMMSS/`

## 6. 建议阅读顺序

- `README.md`
- `docs/how_to_run.md`
- `docs/visualization_guide.md`
- `docs/figure_catalog.md`
- `docs/formula_audit.md`
