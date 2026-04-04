import unittest

import numpy as np

import benchmark_algorithms
from kemm.adapters import BenchmarkPriorAdapter
from kemm.algorithms.kemm import KEMM_DMOEA_Improved
from kemm.benchmark.metrics import PerformanceMetrics
from kemm.benchmark.problems import DynamicTestProblems
from kemm.core.types import KEMMConfig


class CompatibilityExportsTest(unittest.TestCase):
    def test_benchmark_algorithms_exports_new_implementations(self):
        self.assertIs(benchmark_algorithms.KEMM_DMOEA_Improved, KEMM_DMOEA_Improved)
        self.assertIs(benchmark_algorithms.DynamicTestProblems, DynamicTestProblems)
        self.assertIs(benchmark_algorithms.PerformanceMetrics, PerformanceMetrics)

    def test_kemm_supports_benchmark_prior_switch(self):
        algo = KEMM_DMOEA_Improved(
            pop_size=10,
            n_var=4,
            n_obj=2,
            var_bounds=(np.zeros(4), np.ones(4)),
            benchmark_aware_prior=False,
        )
        self.assertFalse(algo.benchmark_aware_prior)

    def test_kemm_accepts_explicit_config_and_adapter(self):
        config = KEMMConfig(
            pop_size=10,
            n_var=4,
            n_obj=2,
            benchmark_aware_prior=True,
            reward_window=7,
        )
        adapter = BenchmarkPriorAdapter()
        algo = KEMM_DMOEA_Improved(
            pop_size=10,
            n_var=4,
            n_obj=2,
            var_bounds=(np.zeros(4), np.ones(4)),
            config=config,
            benchmark_adapter=adapter,
        )
        self.assertTrue(algo.benchmark_aware_prior)
        self.assertEqual(algo.config.reward_window, 7)
        self.assertIs(algo._benchmark_adapter, adapter)


if __name__ == "__main__":
    unittest.main()
