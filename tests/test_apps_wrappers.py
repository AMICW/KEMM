import unittest

import run_experiments
import visualization
from apps import ship_runner
from apps.benchmark_runner import ExperimentConfig as RunnerExperimentConfig
from apps.benchmark_runner import run_benchmark as apps_run_benchmark
from apps.reporting import PerformanceComparisonPlots as NewPerformanceComparisonPlots


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


if __name__ == "__main__":
    unittest.main()
