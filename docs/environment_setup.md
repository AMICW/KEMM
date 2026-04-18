# 环境与安装说明

本文档面向本仓库的实际运行环境搭建，适用于 benchmark 主线与 `ship_simulation` 主线。

建议把它理解成“安装文档”，不是“命令总表”或“实验设计手册”。

如果你已经装好环境，只是忘了命令怎么写，优先看：

- [run_commands.md](run_commands.md)
- [how_to_run.md](how_to_run.md)

如果你想先建立文档地图，再决定往哪看，先回到：

- [README.md](README.md)

## 1. Python 版本

推荐使用：

- Python `3.10` 到 `3.12`

当前代码原则上兼容：

- Python `3.9+`

## 2. 核心依赖

本仓库当前核心第三方依赖较少，主要是：

- `numpy`
- `scipy`
- `matplotlib`

可选依赖：

- `SciencePlots`
- `plotly`

对应安装文件：

- `requirements.txt`
- `requirements-dev.txt`

## 3. 推荐安装方式

### 3.1 使用 `venv`

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
pip install -U pip
pip install -r requirements.txt
```

Linux / macOS:

```bash
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

### 3.2 使用 Conda

```bash
conda create -n kemm python=3.11
conda activate kemm
pip install -r requirements.txt
```

如果你需要测试依赖：

```powershell
pip install -r requirements-dev.txt
```

## 4. 安装后最小验证

安装完成后，建议至少执行下面三步：

```powershell
python -m unittest discover -s tests -v
python -m apps.benchmark_runner --quick
python ship_simulation/run_report.py --quick --scenarios crossing harbor_clutter --n-runs 1
```

这样可以分别验证：

- 基础单元测试是否能跑通
- benchmark 真实入口是否正常
- ship 批量报告真实入口是否正常

如果你要额外验证“最新 ship 报告结构”是否可用（严格可比 + 统计 + 鲁棒性扫描），再补一条：

```powershell
python ship_simulation/run_report.py --quick --summary-only --scenarios crossing --n-runs 1 --algorithms kemm random --strict-comparable --robustness-sweep --robustness-levels 0,0.5 --robustness-scenarios crossing
```

这条命令主要检查新增输出链路是否都能生成：

- `raw/statistical_tests.json/csv`
- `reports/statistical_significance.md`
- `raw/robustness_runs.csv`
- `raw/robustness_curve.csv`
- `raw/robustness_summary.json`
- `reports/robustness_sweep.md`

如果你想验证缓存链路：

```powershell
python -m apps.benchmark_runner --quick --force-rerun
python -m apps.benchmark_runner --quick
```

## 5. Windows 说明

在 Windows 环境下，`joblib/loky` 有时会尝试调用 `wmic` 检测 CPU 核心数，可能出现警告栈。

当前这类警告通常：

- 不影响 benchmark 结果
- 不影响 ship 报告生成
- 不属于算法逻辑错误

如果你想尽量减少这类噪声，可以在当前终端设置：

PowerShell:

```powershell
$env:LOKY_MAX_CPU_COUNT = "1"
```

CMD:

```cmd
set LOKY_MAX_CPU_COUNT=1
```

## 6. 输出目录

benchmark 输出默认写到：

```text
benchmark_outputs/
```

ship 输出默认写到：

```text
ship_simulation/outputs/
```

补充说明：

- benchmark 任务缓存默认写到 `benchmark_outputs/_cache/benchmark_tasks/`
- ship 完整报告的 episode 缓存写到对应报告目录下的 `raw/episode_cache/`

这些目录通常不应作为源码的一部分手工编辑。

## 7. 面向 GitHub 使用者的最小步骤

如果你只是想把仓库拉下来快速确认它可运行：

```bash
git clone <your-repo-url>
cd KEMM(renew)
python -m venv .venv
pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m apps.benchmark_runner --quick
```

如果你还要顺手确认 ship 主线：

```bash
python ship_simulation/run_report.py --quick --scenarios crossing --n-runs 1
```

## 8. 进一步阅读

- `README.md`
- `AGENTS.md`
- `docs/how_to_run.md`
- `docs/run_commands.md`
- `docs/codebase_reference.md`
- `docs/visualization_guide.md`
