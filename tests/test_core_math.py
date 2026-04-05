import unittest

import numpy as np

from kemm.adapters import BenchmarkPriorAdapter
from kemm.algorithms.kemm import KEMM_DMOEA_Improved
from kemm.core.adaptive import AdaptiveOperatorSelector, UCB1Bandit
from kemm.core.drift import LightweightGPR
from kemm.core.memory import VAECompressedMemory
from kemm.core.transfer import GrassmannGeodesicFlow


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
