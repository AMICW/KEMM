import unittest
import shutil
from pathlib import Path
from unittest.mock import patch

import numpy as np

import run_experiments
import visualization
from apps import ship_runner
from apps.benchmark_runner import ExperimentRunner
from apps.benchmark_runner import ExperimentConfig as RunnerExperimentConfig
from apps.benchmark_runner import run_benchmark as apps_run_benchmark
from apps.reporting import PerformanceComparisonPlots as NewPerformanceComparisonPlots
from kemm.algorithms import RI_DMOEA
from kemm.core.types import KEMMConfig


class AppWrapperTests(unittest.TestCase):
    def test_run_experiments_wrapper_reexports_runner_symbols(self):
        self.assertIs(run_experiments.ExperimentConfig, RunnerExperimentConfig)
        self.assertIs(run_experiments.run_benchmark, apps_run_benchmark)

    def test_ship_runner_exposes_demo_entrypoints(self):
        self.assertTrue(callable(ship_runner.run_demo))
        self.assertTrue(callable(ship_runner.generate_report))

    def test_visualization_wrapper_reexports_reporting_symbols(self):
        self.assertIs(visualization.PerformanceComparisonPlots, NewPerformanceComparisonPlots)
        self.assertTrue(callable(visualization.generate_all_figures))

    def test_benchmark_runner_advances_time_with_actual_generations(self):
        class TinyConfig:
            POP_SIZE = 4
            N_VAR = 2
            N_OBJ = 2
            NT = 10
            TAU_T = 5
            N_CHANGES = 3
            GENS_PER_CHANGE = 2
            N_RUNS = 1
            KEMM_CONFIG = None

        class TraceProblems:
            @staticmethod
            def get_time(generation):
                return float(generation)

        class TraceAlgo:
            def __init__(self, pop_size, n_var, n_obj, var_bounds):
                self.pop_size = pop_size
                self.n_var = n_var
                self.n_obj = n_obj
                self.var_bounds = var_bounds
                self.population = None
                self.fitness = None
                self.change_times = []

            def initialize(self):
                self.population = np.zeros((self.pop_size, self.n_var), dtype=float)

            def evaluate(self, pop, obj_func, t):
                self.change_times.append(float(t))
                return np.zeros((len(np.atleast_2d(pop)), self.n_obj), dtype=float)

            def respond_to_change(self, obj_func, t):
                self.change_times.append(float(t))
                self.fitness = np.zeros((self.pop_size, self.n_obj), dtype=float)

            def evolve_one_gen(self, obj_func, t):
                self.fitness = np.zeros((self.pop_size, self.n_obj), dtype=float)

            def get_pareto_front(self):
                return np.zeros((1, self.n_obj), dtype=float)

        def obj_func(population, t):
            population = np.atleast_2d(population)
            return np.zeros((len(population), 2), dtype=float)

        def pof_func(t):
            _ = t
            return np.zeros((1, 2), dtype=float)

        runner = ExperimentRunner(TinyConfig())
        runner.problems = TraceProblems()
        result = runner._run_single(TraceAlgo, obj_func, pof_func)

        self.assertEqual(result["algo_instance"].change_times, [0.0, 2.0, 4.0])

    def test_benchmark_runner_collects_setting_results_and_canonical_slice(self):
        class TinySweepConfig:
            POP_SIZE = 8
            N_VAR = 2
            N_OBJ = 2
            NT = 10
            TAU_T = 10
            SETTINGS = [(5, 10), (10, 10)]
            CANONICAL_SETTING = (10, 10)
            N_CHANGES = 1
            GENS_PER_CHANGE = 1
            N_RUNS = 1
            PROBLEMS = ["FDA1"]
            ALGORITHMS = {"RI": RI_DMOEA}
            ABLATION_VARIANTS = {"KEMM-Full": {"config_overrides": {}}}
            KEMM_CONFIG = KEMMConfig(benchmark_aware_prior=False, memory_online_epochs=1)

        runner = ExperimentRunner(TinySweepConfig())
        results = runner.run_all()

        self.assertEqual(set(runner.setting_results.keys()), {"5,10", "10,10"})
        self.assertEqual(results, runner.setting_results["10,10"])
        self.assertEqual(runner.igd_curves, runner.setting_igd_curves["10,10"])

    def test_benchmark_runner_task_cache_reuses_previous_result(self):
        cache_dir = Path("benchmark_outputs") / f"test_task_cache_{int(np.random.randint(1_000_000))}"

        class TinyCacheConfig:
            POP_SIZE = 4
            N_VAR = 2
            N_OBJ = 2
            NT = 10
            TAU_T = 10
            SETTINGS = [(10, 10)]
            CANONICAL_SETTING = (10, 10)
            N_CHANGES = 1
            GENS_PER_CHANGE = 1
            N_RUNS = 1
            PROBLEMS = ["FDA1"]
            ALGORITHMS = {"RI": RI_DMOEA}
            KEMM_CONFIG = KEMMConfig(benchmark_aware_prior=False, memory_online_epochs=1)
            CACHE_ENABLED = True
            FORCE_RERUN = False
            CACHE_DIR = str(cache_dir)

        runner = ExperimentRunner(TinyCacheConfig())
        calls = {"count": 0}

        def fake_run_single(*args, **kwargs):
            _ = args, kwargs
            calls["count"] += 1
            return {
                "migd": 0.1,
                "sp": 0.2,
                "ms": 0.3,
                "time": 0.4,
                "igd_curve": [0.5],
                "hv_curve": [0.6],
                "change_diagnostics": [],
            }

        runner._run_single = fake_run_single
        runner._run_setting_sweep({"RI": RI_DMOEA}, progress_prefix="TST", collect_curves=False, collect_diagnostics=False)
        self.assertEqual(calls["count"], 1)
        self.assertEqual(runner.task_cache_hits, 0)
        self.assertEqual(runner.task_cache_misses, 1)

        runner._run_setting_sweep({"RI": RI_DMOEA}, progress_prefix="TST", collect_curves=False, collect_diagnostics=False)
        self.assertEqual(calls["count"], 1)
        self.assertEqual(runner.task_cache_hits, 1)
        self.assertEqual(runner.task_cache_misses, 0)

        shutil.rmtree(cache_dir, ignore_errors=True)

    def test_benchmark_runner_skips_problem_aux_precompute_when_cache_is_warm(self):
        cache_dir = Path("benchmark_outputs") / f"test_task_cache_warm_{int(np.random.randint(1_000_000))}"

        class TinyWarmCacheConfig:
            POP_SIZE = 4
            N_VAR = 2
            N_OBJ = 2
            NT = 10
            TAU_T = 10
            SETTINGS = [(10, 10)]
            CANONICAL_SETTING = (10, 10)
            N_CHANGES = 1
            GENS_PER_CHANGE = 1
            N_RUNS = 1
            PROBLEMS = ["FDA1"]
            ALGORITHMS = {"RI": RI_DMOEA}
            KEMM_CONFIG = KEMMConfig(benchmark_aware_prior=False, memory_online_epochs=1)
            CACHE_ENABLED = True
            FORCE_RERUN = False
            CACHE_DIR = str(cache_dir)

        runner = ExperimentRunner(TinyWarmCacheConfig())
        runner._run_single = lambda *args, **kwargs: {
            "migd": 0.1,
            "sp": 0.2,
            "ms": 0.3,
            "time": 0.4,
            "igd_curve": [0.5],
            "hv_curve": [0.6],
            "change_diagnostics": [],
        }
        runner._run_setting_sweep({"RI": RI_DMOEA}, progress_prefix="TST", collect_curves=False, collect_diagnostics=False)

        with patch("apps.benchmark_runner._problem_aux_snapshot", side_effect=AssertionError("problem aux should not be recomputed")):
            runner._run_setting_sweep({"RI": RI_DMOEA}, progress_prefix="TST", collect_curves=False, collect_diagnostics=False)

        shutil.rmtree(cache_dir, ignore_errors=True)

    def test_benchmark_runner_only_keeps_curves_for_canonical_setting(self):
        class TinyCurveConfig:
            POP_SIZE = 4
            N_VAR = 2
            N_OBJ = 2
            NT = 10
            TAU_T = 10
            SETTINGS = [(5, 10), (10, 10)]
            CANONICAL_SETTING = (10, 10)
            N_CHANGES = 1
            GENS_PER_CHANGE = 1
            N_RUNS = 1
            PROBLEMS = ["FDA1"]
            ALGORITHMS = {"RI": RI_DMOEA}
            KEMM_CONFIG = KEMMConfig(benchmark_aware_prior=False, memory_online_epochs=1)

        runner = ExperimentRunner(TinyCurveConfig())
        runner._run_single = lambda *args, **kwargs: {
            "migd": 0.0,
            "sp": 0.0,
            "ms": 0.0,
            "time": 0.0,
            "igd_curve": [0.1],
            "hv_curve": [0.2],
            "change_diagnostics": [],
        }
        _, setting_igd_curves, setting_hv_curves, setting_diagnostics = runner._run_setting_sweep(
            {"RI": RI_DMOEA},
            progress_prefix="TST",
            collect_curves=True,
            collect_diagnostics=True,
        )

        self.assertEqual(set(setting_igd_curves.keys()), {"10,10"})
        self.assertEqual(set(setting_hv_curves.keys()), {"10,10"})
        self.assertEqual(set(setting_diagnostics.keys()), {"10,10"})

    def test_experiment_config_declares_four_module_ablation_variants(self):
        self.assertEqual(
            set(RunnerExperimentConfig.ABLATION_VARIANTS.keys()),
            {
                "KEMM-Full",
                "KEMM-NoMemory",
                "KEMM-NoPrediction",
                "KEMM-NoTransfer",
                "KEMM-NoAdaptive",
            },
        )
        self.assertFalse(RunnerExperimentConfig.ABLATION_BENCHMARK_PRIOR)

    def test_experiment_config_instances_do_not_share_mutable_state(self):
        cfg_a = RunnerExperimentConfig()
        cfg_b = RunnerExperimentConfig()

        cfg_a.PROBLEMS.append("CustomProblem")
        cfg_a.SETTINGS.append((99, 99))
        cfg_a.KEMM_CONFIG.enable_memory = False

        self.assertNotIn("CustomProblem", cfg_b.PROBLEMS)
        self.assertNotIn((99, 99), cfg_b.SETTINGS)
        self.assertTrue(cfg_b.KEMM_CONFIG.enable_memory)

    def test_benchmark_runner_restores_global_numpy_random_state_between_runs(self):
        class TinySeedConfig:
            POP_SIZE = 8
            N_VAR = 2
            N_OBJ = 2
            NT = 10
            TAU_T = 10
            SETTINGS = [(10, 10)]
            CANONICAL_SETTING = (10, 10)
            N_CHANGES = 1
            GENS_PER_CHANGE = 1
            N_RUNS = 1
            PROBLEMS = ["FDA1"]
            ALGORITHMS = {"RI": RI_DMOEA}
            KEMM_CONFIG = KEMMConfig(benchmark_aware_prior=False, memory_online_epochs=1)

        class DummyProblems:
            @staticmethod
            def get_problem(_name):
                return lambda pop, t: np.zeros((len(np.atleast_2d(pop)), 2), dtype=float), lambda t=0.0: np.zeros((1, 2), dtype=float)

        runner = ExperimentRunner(TinySeedConfig())
        runner._run_single = lambda *args, **kwargs: {"migd": 0.0, "sp": 0.0, "ms": 0.0, "time": 0.0, "igd_curve": [], "hv_curve": [], "change_diagnostics": []}

        np.random.seed(12345)
        expected = np.random.rand(5)
        np.random.seed(12345)
        runner.problems = DummyProblems()
        runner._run_setting_sweep({"RI": RI_DMOEA}, progress_prefix="TST", collect_curves=False, collect_diagnostics=False)
        actual = np.random.rand(5)

        np.testing.assert_allclose(actual, expected)

    def test_run_ablation_all_disables_benchmark_prior_by_default(self):
        class TinyAblationConfig:
            POP_SIZE = 8
            N_VAR = 2
            N_OBJ = 2
            NT = 10
            TAU_T = 10
            SETTINGS = [(10, 10)]
            CANONICAL_SETTING = (10, 10)
            N_CHANGES = 1
            GENS_PER_CHANGE = 1
            N_RUNS = 1
            PROBLEMS = ["FDA1"]
            ALGORITHMS = {"RI": RI_DMOEA}
            ABLATION_VARIANTS = {"KEMM-Full": {"config_overrides": {}}}
            ABLATION_BENCHMARK_PRIOR = False
            KEMM_CONFIG = KEMMConfig(benchmark_aware_prior=True, memory_online_epochs=1)

        runner = ExperimentRunner(TinyAblationConfig())
        captured = {}

        def fake_run_setting_sweep(algorithms, **kwargs):
            captured.update(algorithms)
            return {}, {}, {}, {}

        runner._run_setting_sweep = fake_run_setting_sweep
        runner.run_ablation_all()

        self.assertIn("KEMM-Full", captured)
        self.assertFalse(captured["KEMM-Full"]["benchmark_aware_prior"])


if __name__ == "__main__":
    unittest.main()
