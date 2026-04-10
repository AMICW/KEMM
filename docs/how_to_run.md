# How To Run

This file is the English command cheat sheet for the repository.

Use this file when:

- you want a compact English reference for benchmark and ship commands
- you already know the project structure and only need runnable entry points

If you need installation help first, go to [environment_setup.md](environment_setup.md).
If you prefer the Chinese quick-reference version, go to [run_commands.md](run_commands.md).
If you want the full documentation map, go to [README.md](README.md).

## 1. Real Entry Points

Benchmark real entry point:

- `python -m apps.benchmark_runner`

Ship physical batch-report real entry point:

- `python ship_simulation/run_report.py`

These two commands are the primary entry points of the current codebase.

## 2. Benchmark

### 2.1 Quick smoke test

```powershell
python -m apps.benchmark_runner --quick --plot-preset paper
```

Current `--quick` meaning:

- runs: `2`
- changes: `5`
- generations per change: `10`
- problems: `FDA1`, `FDA3`, `dMOP2`

### 2.2 Full benchmark

```powershell
python -m apps.benchmark_runner --full --plot-preset paper
```

Current `--full` meaning:

- runs: `5`
- changes: `10`
- generations per change: `20`
- problems: `FDA1`, `FDA2`, `FDA3`, `dMOP1`, `dMOP2`, `dMOP3`
- settings sweep: `(5,10)`, `(10,10)`, `(10,20)`

### 2.3 Full benchmark with JY problems

```powershell
python -m apps.benchmark_runner --with-jy --plot-preset paper
```

This appends `JY1` and `JY4` to the standard problem suite.

### 2.4 Medium benchmark

There is no built-in `--medium` CLI switch.

Use the public API with a one-off PowerShell script:

```powershell
@'
from pathlib import Path
from apps.benchmark_runner import ExperimentConfig, ExperimentRunner, ResultPresenter
from kemm.reporting import build_report_paths, export_benchmark_report
from reporting_config import build_benchmark_plot_config

cfg = ExperimentConfig()
cfg.N_RUNS = 3
cfg.N_CHANGES = 7
cfg.GENS_PER_CHANGE = 15
cfg.PROBLEMS = ["FDA1", "FDA2", "FDA3", "dMOP1", "dMOP2", "dMOP3"]

report_root = build_report_paths(Path("benchmark_outputs/medium"), prefix="benchmark")
(report_root / "figures").mkdir(parents=True, exist_ok=True)

runner = ExperimentRunner(cfg)
results = runner.run_all()
ablation_results = runner.run_ablation_all()

plot_config = build_benchmark_plot_config("paper")
presenter = ResultPresenter(
    results,
    cfg,
    igd_curves=runner.igd_curves,
    hv_curves=runner.hv_curves,
    algorithm_diagnostics=runner.algorithm_diagnostics,
    ablation_results=ablation_results,
    setting_results=runner.setting_results,
    ablation_setting_results=runner.ablation_setting_results,
)
presenter.plot_all(prefix=str(report_root / "figures" / "benchmark"), plot_config=plot_config)
export_benchmark_report(
    results,
    cfg,
    output_root=report_root,
    ablation_results=ablation_results,
    setting_results=runner.setting_results,
    ablation_setting_results=runner.ablation_setting_results,
)
print(report_root)
'@ | python -
```

Recommended medium-scale values in this script:

- `N_RUNS = 3`
- `N_CHANGES = 7`
- `GENS_PER_CHANGE = 15`

### 2.5 Useful benchmark options

Custom output directory:

```powershell
python -m apps.benchmark_runner --full --output-dir benchmark_outputs\my_run
```

Different plot presets:

```powershell
python -m apps.benchmark_runner --full --plot-preset ieee
python -m apps.benchmark_runner --full --plot-preset nature
python -m apps.benchmark_runner --full --plot-preset thesis
```

Appendix plots and interactive figure bundles:

```powershell
python -m apps.benchmark_runner --full --appendix-plots --interactive-figures
```

### 2.6 Benchmark outputs you should read first

After a full benchmark run, the most important outputs are:

- `reports/summary.md`
- `raw/summary.json`
- `raw/paper_table_metrics.csv`
- `raw/ablation_delta_metrics.csv`
- `figures/benchmark_migd_table.png`
- `figures/benchmark_ablation.png`

Interpretation notes:

- `benchmark_migd_table.png` is the paper-style main table across
  `(n_t, tau_t) = (5,10), (10,10), (10,20)`.
- `benchmark_ablation.png` plots relative `MIGD` degradation versus
  `KEMM-Full`, so positive values mean the ablated variant is worse.
- If you ever see a historical report where `FDA1` and `dMOP3` have identical
  metric rows, treat it as stale output generated before the `dMOP3` fix.

### 2.7 Compatibility wrapper

Legacy wrapper commands still work:

```powershell
python run_experiments.py --quick
python run_experiments.py --full
```

But the real entry point is still:

```powershell
python -m apps.benchmark_runner
```

## 3. Ship Physical Simulation

### 3.1 Important distinction

This command:

```powershell
python -m apps.ship_runner
```

is only a single demo entry point.

This command:

```powershell
python ship_simulation/run_report.py
```

is the full ship physical batch-report entry point.

If you want the complete ship physical test module, use:

```powershell
python ship_simulation/run_report.py
```

### 3.2 Single demo run

Default demo:

```powershell
python -m apps.ship_runner
```

Single scenario, explicit optimizer:

```powershell
python -c "from ship_simulation.main_demo import run_demo; run_demo('crossing', optimizer_name='kemm', show_animation=False)"
python -c "from ship_simulation.main_demo import run_demo; run_demo('harbor_clutter', optimizer_name='kemm', show_animation=False)"
```

Use this when you want:

- one scenario only
- no full batch report
- a quick run for debugging

### 3.3 Quick ship report

```powershell
python ship_simulation/run_report.py --quick --scenarios crossing --n-runs 1 --plot-preset paper
```

Two key scenarios:

```powershell
python ship_simulation/run_report.py --quick --scenarios crossing harbor_clutter --n-runs 1 --plot-preset paper
```

Current `--quick` meaning:

- `random_search_samples = 20`
- `NSGA-style pop_size = 22`
- `NSGA-style generations = 10`
- `KEMM pop_size = 28`
- `KEMM generations = 12`
- `n_runs = 1`

### 3.4 Full ship physical test

This is the most important ship command:

```powershell
python ship_simulation/run_report.py
```

By default it runs:

- scenarios: `head_on`, `crossing`, `overtaking`, `harbor_clutter`
- optimizers: `kemm`, `nsga_style`, `random`
- repeated runs per optimizer per scenario: `3`

Default important values:

- `KEMM pop_size = 48`
- `KEMM generations = 24`
- `NSGA-style pop_size = 36`
- `NSGA-style generations = 16`
- `Random search samples = 48`
- `n_runs = 3`

### 3.5 Select scenarios and repeated runs

Only the dense harbor scenario:

```powershell
python ship_simulation/run_report.py --scenarios harbor_clutter --n-runs 3 --plot-preset paper
```

Explicit full scenario list:

```powershell
python ship_simulation/run_report.py --scenarios head_on crossing overtaking harbor_clutter --n-runs 3 --plot-preset paper
```

### 3.6 Interactive 3D export

Ship rule summary:

- only 3D plots get interactive exports
- 2D plots stay PNG-only

Command:

```powershell
python ship_simulation/run_report.py --scenarios head_on crossing overtaking harbor_clutter --n-runs 3 --plot-preset paper --interactive-figures --interactive-html
```

### 3.6b Run dynamic ship experiment profiles

Use `--experiment-profile` when you want the rolling-horizon ship report to include scheduled scenario changes.

Command:

```powershell
python ship_simulation/run_report.py --scenarios harbor_clutter --n-runs 3 --plot-preset paper --experiment-profile drift
python ship_simulation/run_report.py --scenarios harbor_clutter --n-runs 3 --plot-preset paper --experiment-profile shock
python ship_simulation/run_report.py --scenarios harbor_clutter --n-runs 3 --plot-preset paper --experiment-profile recurring_harbor
```

Supported profiles:

- `baseline`: no extra scheduled changes
- `drift`: gradual environment and traffic drift
- `shock`: sudden closure and stronger perturbation
- `recurring_harbor`: harbor drift followed by partial recovery

### 3.7 Adjust scenario-generation parameters

If you want to tune ship scenarios without editing `generator.py`, use:

- `ProblemConfig.scenario_generation` in `ship_simulation/config.py`

Example:

```python
from ship_simulation.config import build_default_config

config = build_default_config()
config.scenario_generation.harbor_clutter.circular_obstacle_limit = 5
config.scenario_generation.harbor_clutter.polygon_obstacle_limit = 2
config.scenario_generation.harbor_clutter.target_limit = 2
config.scenario_generation.harbor_clutter.circular_radius_scale = 0.55
config.scenario_generation.harbor_clutter.channel_width_scale = 1.20
```

This lets you directly control:

- obstacle counts
- traffic-ship counts
- obstacle scales
- channel width
- scalar/vector field intensity

If you want to control dynamic ship experiments in code instead of CLI, use:

```python
from ship_simulation.config import apply_experiment_profile, build_default_config

config = build_default_config()
apply_experiment_profile(config, "shock")
```

This writes the scheduled changes into `ProblemConfig.experiment`.

## 4. Output Directories

Benchmark default output:

```text
benchmark_outputs/benchmark_YYYYMMDD_HHMMSS/
```

Ship default output:

```text
ship_simulation/outputs/report_YYYYMMDD_HHMMSS/
```

The main subdirectories are:

- `figures/`
- `raw/`
- `reports/`

## 5. Tests

Full unit test suite:

```powershell
python -m unittest discover -s tests -v
```

Quick benchmark regression:

```powershell
python -m apps.benchmark_runner --quick
```

Quick ship regression:

```powershell
python ship_simulation/run_report.py --quick --scenarios crossing harbor_clutter --n-runs 1
```

## 6. One-Screen Cheat Sheet

Benchmark quick:

```powershell
python -m apps.benchmark_runner --quick --plot-preset paper
```

Benchmark medium:

```text
Use the PowerShell script in section 2.4 of this file.
```

Benchmark full:

```powershell
python -m apps.benchmark_runner --full --plot-preset paper
```

Ship single demo:

```powershell
python -m apps.ship_runner
```

Ship single harbor run:

```powershell
python -c "from ship_simulation.main_demo import run_demo; run_demo('harbor_clutter', optimizer_name='kemm', show_animation=False)"
```

Ship quick report:

```powershell
python ship_simulation/run_report.py --quick --scenarios crossing harbor_clutter --n-runs 1 --plot-preset paper
```

Ship full physical test:

```powershell
python ship_simulation/run_report.py
```
