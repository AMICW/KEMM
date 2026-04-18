# How To Run

This file is the English command cheat sheet for the repository. It is meant for people who already have the environment set up and mainly need the current runnable entry points and the most useful options.

If you need installation help first, go to [environment_setup.md](environment_setup.md).
If you prefer the Chinese quick-reference version, go to [run_commands.md](run_commands.md).
If you want the full documentation map, go to [README.md](README.md).

## 1. Real Entry Points

Benchmark real entry point:

- `python -m apps.benchmark_runner`

Ship batch-report real entry point:

- `python ship_simulation/run_report.py`

Compatibility wrappers still exist, but they are no longer the preferred entry points:

- `python run_experiments.py`
- `python -m apps.ship_runner`

---

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
- default workers: `1`

### 2.2 Full benchmark

README keeps only this full-run entry point:

```powershell
python -m apps.benchmark_runner --full --workers 4
```

Current `--full` meaning:

- runs: `5`
- changes: `10`
- generations per change: `20`
- problems: `FDA1`, `FDA2`, `FDA3`, `dMOP1`, `dMOP2`, `dMOP3`
- settings sweep: `(5,10)`, `(10,10)`, `(10,20)`
- ablation and main figures enabled by default
- benchmark task cache enabled by default

### 2.3 Full benchmark with JY problems

```powershell
python -m apps.benchmark_runner --with-jy --workers 4
```

This appends `JY1` and `JY4` to the standard problem suite.

### 2.4 Benchmark cache and warm reruns

The benchmark runner now uses task-level caching for complete runs. The cache granularity is:

- `(setting, algorithm, problem, run, ablation_variant)`

Cache directory:

- `benchmark_outputs/_cache/benchmark_tasks/`

If you rerun the same configuration, cached tasks are reused automatically.

Force a full recomputation:

```powershell
python -m apps.benchmark_runner --full --force-rerun
```

A simple warm-rerun check:

```powershell
python -m apps.benchmark_runner --quick --force-rerun
python -m apps.benchmark_runner --quick
```

### 2.5 Useful benchmark options

Skip ablation variants:

```powershell
python -m apps.benchmark_runner --full --skip-ablation
```

Export raw tables and Markdown summary only, without rendering figures:

```powershell
python -m apps.benchmark_runner --full --summary-only
```

Restrict algorithms or problems:

```powershell
python -m apps.benchmark_runner --quick --algorithms KEMM RI Tr
python -m apps.benchmark_runner --quick --problems FDA1 FDA3 dMOP2
```

Custom output directory:

```powershell
python -m apps.benchmark_runner --full --output-dir benchmark_outputs\my_run
```

Plot presets and SciencePlots override:

```powershell
python -m apps.benchmark_runner --full --plot-preset ieee
python -m apps.benchmark_runner --full --science-style science,ieee,no-latex
```

Appendix plots and interactive figure bundles:

```powershell
python -m apps.benchmark_runner --full --appendix-plots --interactive-figures
```

### 2.6 Benchmark outputs to read first

After a complete run, read these first:

- `reports/summary.md`
- `raw/summary.json`
- `raw/paper_table_metrics.csv`
- `raw/ablation_delta_metrics.csv`
- `figures/benchmark_migd_table.png`
- `figures/benchmark_ablation.png`

Notes:

- `benchmark_migd_table.png` is the paper-style main table across the three dynamic settings.
- `benchmark_ablation.png` plots relative `MIGD` degradation versus `KEMM-Full`, so positive values mean the ablated variant is worse.
- Only the canonical setting keeps the full `igd_curve / hv_curve / change_diagnostics` aggregation; non-canonical settings mainly keep scalar end metrics.

### 2.7 Compatibility wrapper

```powershell
python run_experiments.py --quick
python run_experiments.py --full
```

These commands still work, but the real entry point remains:

```powershell
python -m apps.benchmark_runner
```

---

## 3. Ship Physical Simulation

### 3.1 Important distinction

Single demo entry point:

```powershell
python -m apps.ship_runner
```

Full batch-report entry point:

```powershell
python ship_simulation/run_report.py
```

Use the second command when you want the full ship experiment/report pipeline.

### 3.2 Single demo run

```powershell
python -m apps.ship_runner
python -c "from ship_simulation.main_demo import run_demo; run_demo('crossing', optimizer_name='kemm', show_animation=False)"
python -c "from ship_simulation.main_demo import run_demo; run_demo('harbor_clutter', optimizer_name='kemm', show_animation=False)"
```

This is useful when you want:

- one scenario only
- a quick trajectory check
- no full report export

### 3.3 Quick ship report

```powershell
python ship_simulation/run_report.py --quick --scenarios crossing --n-runs 1 --plot-preset paper
python ship_simulation/run_report.py --quick --scenarios crossing harbor_clutter --n-runs 1 --plot-preset paper
```

Current `--quick` meaning:

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

### 3.4 Full ship report

README keeps only this complete-run command:

```powershell
python ship_simulation/run_report.py --workers 4
```

By default it runs:

- scenarios: `head_on`, `crossing`, `overtaking`, `harbor_clutter`
- optimizers: `kemm`, `nsga_style`, `random`
- repeated runs per optimizer per scenario: `3`
- total episodes: `36`
- default figure count: `4 * 19 + 2 = 78`

Complete mode also enables two important defaults:

- `scenario_profiles.active_profile_name = full_tuned`
- `episode_cache_enabled = True`

The internal execution is now staged as:

1. compute all episodes and write `raw/episode_cache/`
2. render figures in parallel
3. write summary, metadata, and figure inventory

### 3.5 Ship cache and metadata

Episode cache directory:

- `ship_simulation/outputs/report_YYYYMMDD_HHMMSS/raw/episode_cache/`

If you rerun the same configuration against the same output directory:

- episode results are reused from cache
- figures are regenerated completely
- summary and metadata are rewritten

Current metadata includes:

- `scenario_solve_profile`
- `episode_compute_seconds`
- `figure_render_seconds`
- `episode_cache_hits`
- `episode_cache_misses`

### 3.6 Select scenarios, algorithms, and repeated runs

Only the dense harbor scenario:

```powershell
python ship_simulation/run_report.py --scenarios harbor_clutter --n-runs 3 --plot-preset paper
```

Compare only `kemm` and `random`:

```powershell
python ship_simulation/run_report.py --algorithms kemm random --scenarios crossing overtaking
```

Add strict budget-matched baselines (`*_matched`) while keeping the original groups:

```powershell
python ship_simulation/run_report.py --algorithms kemm random nsga_style --strict-comparable
```

Explicit full scenario list:

```powershell
python ship_simulation/run_report.py --scenarios head_on crossing overtaking harbor_clutter --n-runs 3 --plot-preset paper
```

### 3.7 Dynamic experiment profiles

`--experiment-profile` controls scheduled changes during rolling-horizon execution. It is not the same as the scenario solve profile set.

Supported values:

- `baseline`
- `drift`
- `shock`
- `recurring_harbor`

Examples:

```powershell
python ship_simulation/run_report.py --scenarios harbor_clutter --n-runs 3 --experiment-profile drift
python ship_simulation/run_report.py --scenarios harbor_clutter --n-runs 3 --experiment-profile shock
python ship_simulation/run_report.py --scenarios harbor_clutter --n-runs 3 --experiment-profile recurring_harbor
```

Meaning:

- `baseline`: no additional scheduled changes
- `drift`: gradual environment and traffic drift
- `shock`: sudden closure and stronger perturbation
- `recurring_harbor`: harbor drift followed by partial recovery

### 3.8 Solve profile defaults

Ship experiments now use a second profile system through `DemoConfig.scenario_profiles`.

Current defaults:

- complete runs: `full_tuned`
- quick runs: `legacy_uniform`

`full_tuned` changes these values per scenario:

- solver budgets
- local and execution horizon
- `safety_clearance`
- risk-related penalties and `domain_risk_weight`
- `objective_weights`

There is currently no dedicated CLI flag to switch solve profiles. If you want a `legacy_uniform` versus `full_tuned` A/B, set `demo.scenario_profiles.active_profile_name` in Python before calling the report generator.

### 3.9 Statistical significance export

Ship report now exports confidence intervals and pairwise significance tests by default:

- `raw/statistical_tests.json`
- `raw/statistical_tests.csv`
- `reports/statistical_significance.md`

The test policy is:

- Welch t-test when both groups pass normality checks and sample size is large enough
- Mann-Whitney U otherwise

### 3.10 Robustness sweep

Run disturbance-level robustness sweeps and export success-rate curves:

```powershell
python ship_simulation/run_report.py --robustness-sweep --robustness-levels 0,0.25,0.5,0.75,1.0 --robustness-scenarios crossing overtaking harbor_clutter
```

Outputs:

- `raw/robustness_runs.csv`
- `raw/robustness_curve.csv`
- `raw/robustness_summary.json`
- `reports/robustness_sweep.md`
- `figures/robustness_success_curve.png` (when rendering is enabled)

### 3.11 Interactive exports

Only `pareto3d` and `spatiotemporal` produce extra interactive artifacts.

```powershell
python ship_simulation/run_report.py --quick --scenarios crossing --interactive-figures --interactive-html
```

Notes:

- `*.fig.pickle`: matplotlib figure bundle
- `.html`: browser-based interactive export
- other 2D plots remain PNG-only

### 3.12 Useful ship options

Export raw outputs and Markdown summary only, without figures:

```powershell
python ship_simulation/run_report.py --summary-only
```

Plot presets and SciencePlots override:

```powershell
python ship_simulation/run_report.py --plot-preset ieee
python ship_simulation/run_report.py --science-style science,ieee,no-latex
```

Appendix plots:

```powershell
python ship_simulation/run_report.py --appendix-plots
```

### 3.13 Latest-structure smoke check

If you want one command to validate strict comparable + statistical export + robustness export together:

```powershell
python ship_simulation/run_report.py --quick --summary-only --scenarios crossing --n-runs 1 --algorithms kemm random --strict-comparable --robustness-sweep --robustness-levels 0,0.5 --robustness-scenarios crossing
```

Expected key outputs:

- `raw/statistical_tests.json`
- `raw/statistical_tests.csv`
- `reports/statistical_significance.md`
- `raw/robustness_runs.csv`
- `raw/robustness_curve.csv`
- `raw/robustness_summary.json`
- `reports/robustness_sweep.md`

---

## 4. Output Directories

Benchmark:

- `benchmark_outputs/benchmark_YYYYMMDD_HHMMSS/`

Ship:

- `ship_simulation/outputs/report_YYYYMMDD_HHMMSS/`

Shared main subdirectories:

- `figures/`
- `raw/`
- `reports/`

Ship `raw/` files worth checking first:

- `report_metadata.json`
- `figure_manifest.json`
- `representative_runs.json`
- `planning_steps.json`
- `scenario_catalog.json`

---

## 5. Tests and Regressions

Full unit test suite:

```powershell
python -m unittest discover -s tests -v
```

Benchmark quick regression:

```powershell
python -m apps.benchmark_runner --quick --force-rerun
python -m apps.benchmark_runner --quick
```

Ship quick regression:

```powershell
python ship_simulation/run_report.py --quick --scenarios crossing harbor_clutter --n-runs 1
```

---

## 6. Suggested Reading Order

- [README.md](README.md)
- [run_commands.md](run_commands.md)
- [ship_experiment_playbook.md](ship_experiment_playbook.md)
- [visualization_guide.md](visualization_guide.md)
- [figure_catalog.md](figure_catalog.md)
