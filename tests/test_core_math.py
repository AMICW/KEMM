import unittest
from unittest.mock import Mock, patch

import numpy as np

from kemm.algorithms.baselines import MMTL_DMOEA
from kemm.adapters import BenchmarkPriorAdapter
from kemm.algorithms.kemm import KEMM_DMOEA_Improved
from kemm.benchmark.problems import DynamicTestProblems
from kemm.core.adaptive import AdaptiveOperatorSelector, UCB1Bandit
from kemm.core.drift import LightweightGPR
from kemm.core.memory import VAECompressedMemory
from kemm.core.transfer import GrassmannGeodesicFlow, ManifoldTransferLearning
from kemm.core.types import KEMMConfig


class CoreMathSmokeTests(unittest.TestCase):
    def test_ucb1_formula_matches_expected_value(self):
        bandit = UCB1Bandit(n_arms=2, window=5, c=0.5)
        bandit.reward_windows[0].clear()
        bandit.reward_windows[1].clear()
        bandit.reward_windows[0].extend([0.2, 0.4])
        bandit.reward_windows[1].extend([0.5, 0.5, 0.5])
        bandit.counts = np.array([2.0, 4.0])
        bandit.total_count = 6.0

        ucb = bandit.get_ucb_values()
        expected = np.array(
            [
                0.3 + 0.5 * np.sqrt(np.log(7.0) / 2.0),
                0.5 + 0.5 * np.sqrt(np.log(7.0) / 4.0),
            ]
        )
        np.testing.assert_allclose(ucb, expected, rtol=1e-7, atol=1e-7)

    def test_lightweight_gpr_interpolates_simple_signal(self):
        X = np.array([[0.0], [1.0], [2.0]])
        y = np.array([0.0, 1.0, 2.0])
        gp = LightweightGPR(lengthscale=1.0, signal_var=1.0, noise_var=1e-6)
        gp.fit(X, y)
        mu, std = gp.predict(np.array([[1.0]]), return_std=True)
        self.assertAlmostEqual(float(mu[0]), 1.0, places=2)
        self.assertGreaterEqual(float(std[0]), 0.0)

    def test_grassmann_bases_remain_orthonormal(self):
        flow = GrassmannGeodesicFlow(n_subspaces=3)
        PS = np.eye(3, 2)
        PT = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=float)
        bases = flow.compute_geodesic_bases(PS, PT)
        self.assertEqual(len(bases), 3)
        for basis in bases:
            gram = basis.T @ basis
            np.testing.assert_allclose(gram, np.eye(2), atol=1e-5)

    def test_vae_memory_store_and_retrieve(self):
        rng = np.random.default_rng(0)
        memory = VAECompressedMemory(
            input_dim=4,
            latent_dim=2,
            hidden_dim=8,
            capacity=3,
            beta=0.1,
            online_epochs=1,
        )
        solutions = rng.normal(size=(6, 4))
        fitness = rng.normal(size=(6, 2))
        fingerprint = np.linspace(0.0, 1.0, 12)

        memory.store(solutions, fitness, fingerprint)
        result = memory.retrieve(fingerprint, top_k=1, n_decode=4)

        self.assertEqual(len(memory.memory), 1)
        self.assertEqual(len(result), 1)
        self.assertIn("solutions", result[0])
        self.assertIn("similarity", result[0])

    def test_kemm_change_response_preserves_population_shape(self):
        def obj_func(population: np.ndarray, t: float) -> np.ndarray:
            population = np.atleast_2d(population)
            return np.column_stack(
                [
                    np.sum(population**2, axis=1),
                    np.sum((population - 0.5) ** 2, axis=1),
                ]
            )

        algo = KEMM_DMOEA_Improved(
            pop_size=12,
            n_var=4,
            n_obj=2,
            var_bounds=(np.zeros(4), np.ones(4)),
            benchmark_aware_prior=False,
        )
        algo.initialize()
        algo.fitness = algo.evaluate(algo.population, obj_func, 0.0)
        algo.respond_to_change(obj_func, 0.1)

        self.assertEqual(algo.population.shape, (12, 4))
        self.assertEqual(algo.fitness.shape, (12, 2))
        self.assertIsNotNone(algo.get_last_change_diagnostics())
        self.assertGreaterEqual(algo.get_last_change_diagnostics().candidate_pool_size, 12)

    def test_kemm_module_switches_disable_requested_candidate_sources(self):
        def obj_func(population: np.ndarray, t: float) -> np.ndarray:
            population = np.atleast_2d(population)
            return np.column_stack(
                [
                    np.sum(population**2, axis=1),
                    np.sum((population - 0.5) ** 2, axis=1),
                ]
            )

        for field_name, candidate_name in [
            ("enable_memory", "memory"),
            ("enable_prediction", "prediction"),
            ("enable_transfer", "transfer"),
        ]:
            config = KEMMConfig(
                benchmark_aware_prior=False,
                memory_online_epochs=1,
                **{field_name: False},
            )
            algo = KEMM_DMOEA_Improved(
                pop_size=12,
                n_var=4,
                n_obj=2,
                var_bounds=(np.zeros(4), np.ones(4)),
                config=config,
            )
            algo.initialize()
            algo.fitness = algo.evaluate(algo.population, obj_func, 0.0)
            algo.respond_to_change(obj_func, 0.1)
            diagnostics = algo.get_last_change_diagnostics()
            self.assertEqual(diagnostics.actual_counts[candidate_name], 0)
            self.assertEqual(diagnostics.requested_counts[candidate_name], 0)

    def test_kemm_no_adaptive_uses_fixed_equal_ratios(self):
        config = KEMMConfig(
            benchmark_aware_prior=False,
            enable_adaptive=False,
            enable_memory=True,
            enable_prediction=False,
            enable_transfer=True,
        )
        algo = KEMM_DMOEA_Improved(
            pop_size=10,
            n_var=3,
            n_obj=2,
            var_bounds=(np.zeros(3), np.ones(3)),
            config=config,
        )
        ratios = algo._resolve_operator_ratios()
        np.testing.assert_allclose(ratios, np.array([1.0 / 3.0, 0.0, 1.0 / 3.0, 1.0 / 3.0]), atol=1e-8)

    def test_mmtl_archive_respects_fifo_individual_capacity(self):
        algo = MMTL_DMOEA(pop_size=4, n_var=2, n_obj=2, var_bounds=(np.zeros(2), np.ones(2)))
        algo.archive_capacity = 6
        first = np.arange(8, dtype=float).reshape(4, 2)
        second = np.arange(8, 16, dtype=float).reshape(4, 2)
        algo._store_solutions(first)
        algo._store_solutions(second)

        self.assertEqual(algo.archive.shape, (6, 2))
        np.testing.assert_array_equal(algo.archive, np.vstack([first, second])[-6:])

    def test_mmtl_find_best_sol_returns_half_population(self):
        def obj_func(population: np.ndarray, t: float) -> np.ndarray:
            population = np.atleast_2d(population)
            return np.column_stack(
                [
                    np.sum(population**2, axis=1),
                    np.sum((population - 0.2) ** 2, axis=1),
                ]
            )

        algo = MMTL_DMOEA(pop_size=10, n_var=3, n_obj=2, var_bounds=(np.zeros(3), np.ones(3)))
        algo._store_solutions(np.random.uniform(0.0, 1.0, (10, 3)))
        selected = algo._find_best_sol(obj_func, 0.2)

        self.assertEqual(selected.shape, (5, 3))
        self.assertTrue(np.all(selected >= 0.0))
        self.assertTrue(np.all(selected <= 1.0))

    def test_mmtl_transfer_uses_geodesic_transfer_module(self):
        algo = MMTL_DMOEA(pop_size=10, n_var=3, n_obj=2, var_bounds=(np.zeros(3), np.ones(3)))
        last_best = np.random.uniform(0.0, 1.0, (5, 3))
        mock_transfer = Mock(return_value=np.full((5, 3), 0.25))
        algo.transfer_module.transfer = mock_transfer

        transferred = algo._transfer(last_best, lambda pop, t: np.zeros((len(np.atleast_2d(pop)), 2)), 0.1)

        mock_transfer.assert_called_once()
        self.assertEqual(transferred.shape, (5, 3))
        self.assertTrue(np.allclose(transferred, 0.25))

    def test_mmtl_uses_paper_style_transfer_dim(self):
        algo = MMTL_DMOEA(pop_size=10, n_var=4, n_obj=2, var_bounds=(np.zeros(4), np.ones(4)))

        self.assertEqual(type(algo.transfer_module).__name__, "_MMTLTransferLearning")
        self.assertEqual(algo.transfer_module.transfer_dim, 1)
        self.assertEqual(algo.transfer_module._adaptive_dim(None, 4), 1)

    def test_mmtl_initializes_moead_state_and_evolves(self):
        def obj_func(population: np.ndarray, t: float) -> np.ndarray:
            population = np.atleast_2d(population)
            return np.column_stack(
                [
                    np.sum(population**2, axis=1),
                    np.sum((population - 0.3) ** 2, axis=1),
                ]
            )

        algo = MMTL_DMOEA(pop_size=12, n_var=4, n_obj=2, var_bounds=(np.zeros(4), np.ones(4)))
        algo.initialize()
        algo.fitness = algo.evaluate(algo.population, obj_func, 0.0)
        algo.evolve_one_gen(obj_func, 0.0)

        self.assertEqual(algo.weight_vectors.shape, (12, 2))
        self.assertEqual(algo.neighbors.shape[0], 12)
        self.assertEqual(algo.population.shape, (12, 4))
        self.assertEqual(algo.fitness.shape, (12, 2))
        self.assertIsNotNone(algo.ideal_point)

    def test_mmtl_find_best_sol_falls_back_without_sklearn(self):
        def obj_func(population: np.ndarray, t: float) -> np.ndarray:
            population = np.atleast_2d(population)
            return np.column_stack(
                [
                    np.sum(population**2, axis=1),
                    np.sum((population - 0.4) ** 2, axis=1),
                ]
            )

        algo = MMTL_DMOEA(pop_size=8, n_var=3, n_obj=2, var_bounds=(np.zeros(3), np.ones(3)))
        algo._store_solutions(np.random.uniform(0.0, 1.0, (8, 3)))
        with patch("kemm.algorithms.baselines.HAS_SKLEARN", False):
            selected = algo._find_best_sol(obj_func, 0.1)
        self.assertEqual(selected.shape, (4, 3))

    def test_manifold_transfer_supports_one_dimensional_subspace(self):
        transfer = ManifoldTransferLearning(n_clusters=1, n_subspaces=3)
        source = np.array(
            [
                [0.10, 0.10, 0.00],
                [0.20, 0.20, 0.00],
                [0.30, 0.30, 0.00],
                [0.40, 0.40, 0.00],
            ],
            dtype=float,
        )
        target = np.array(
            [
                [0.10, 0.00, 0.10],
                [0.20, 0.00, 0.20],
                [0.30, 0.00, 0.30],
                [0.40, 0.00, 0.40],
                [0.50, 0.00, 0.50],
            ],
            dtype=float,
        )

        transferred = transfer.transfer(source, target, transfer_size=4)

        self.assertEqual(transferred.shape, (4, 3))
        self.assertFalse(np.allclose(transferred[:, 2], 0.0))

    def test_benchmark_prior_adapter_generates_problem_aware_samples(self):
        def fda1(population: np.ndarray, t: float) -> np.ndarray:
            population = np.atleast_2d(population)
            return np.column_stack([population[:, 0], 1.0 - np.sqrt(population[:, 0])])

        adapter = BenchmarkPriorAdapter()
        samples = adapter.generate(
            obj_func=fda1,
            t=0.25,
            n_samples=6,
            lb=np.array([0.0, -1.0, -1.0, -1.0]),
            ub=np.array([1.0, 1.0, 1.0, 1.0]),
            n_var=4,
        )
        self.assertIsNotNone(samples)
        self.assertEqual(samples.shape, (6, 4))

    def test_dmop3_uses_time_varying_active_coordinate(self):
        problems = DynamicTestProblems(nt=10, tau_t=10)
        x = np.array(
            [
                [0.10, 0.90, 0.25, 0.60],
                [0.80, 0.20, 0.75, 0.10],
            ],
            dtype=float,
        )
        fda1_values = problems.fda1(x, t=0.1)
        dmop3_values = problems.dmop3(x, t=0.1)

        self.assertEqual(problems._dmop3_active_index(0.0, 4), 0)
        self.assertEqual(problems._dmop3_active_index(0.1, 4), 1)
        self.assertEqual(problems._dmop3_active_index(0.2, 4), 2)
        self.assertFalse(np.allclose(dmop3_values, fda1_values))
        np.testing.assert_allclose(dmop3_values[:, 0], x[:, 1], atol=1e-12)

    def test_selector_quality_update_rewards_higher_quality(self):
        selector = AdaptiveOperatorSelector()
        selector._prev_ratios = np.array([0.6, 0.2, 0.1, 0.1], dtype=float)
        selector.update_with_quality(0.2)
        before = selector.bandit.counts.copy()
        selector.update_with_quality(0.5)
        after = selector.bandit.counts.copy()

        self.assertGreater(after[0], before[0])
        self.assertGreater(selector.bandit.rewards_history[-1], 0.0)


if __name__ == "__main__":
    unittest.main()
