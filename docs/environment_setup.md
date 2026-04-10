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

本仓库当前核心第三方依赖很少，主要是：

- `numpy`
- `scipy`
- `matplotlib`

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

## 4. 验证安装

安装完成后，建议至少执行以下命令：

```bash
python -m unittest discover -s tests -v
python run_experiments.py --quick
python ship_simulation/run_report.py
```

如果你需要更完整的运行命令总表，包括：

- benchmark 快速 / 中等 / 完整实验
- ship 单次 demo / 快速报告 / 完整物理测试
- 输出目录与常用可选参数

请直接看：

- `docs/how_to_run.md`
- `docs/run_commands.md`

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

## 6. 仓库输出目录

benchmark 输出默认写到：

```text
benchmark_outputs/
```

ship 输出默认写到：

```text
ship_simulation/outputs/
```

这些目录通常不应作为源码的一部分手工编辑。

## 7. 面向 GitHub 使用者的最小步骤

如果你只是想把仓库拉下来快速确认它可运行：

```bash
git clone <your-repo-url>
cd KEMM(renew)
python -m venv .venv
pip install -r requirements.txt
python -m unittest discover -s tests -v
python run_experiments.py --quick
```

## 8. 进一步阅读

- `README.md`
- `AGENTS.md`
- `docs/how_to_run.md`
- `docs/run_commands.md`
- `docs/codebase_reference.md`
- `docs/visualization_guide.md`
