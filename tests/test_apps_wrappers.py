import unittest

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
