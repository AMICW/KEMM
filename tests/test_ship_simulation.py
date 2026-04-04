import os
import unittest
from pathlib import Path

import matplotlib
import numpy as np

from reporting_config import PublicationStyle, ShipPlotConfig
from ship_simulation.config import DemoConfig, KEMMConfig, build_default_config
from ship_simulation.optimizer.interface import ShipOptimizerInterface
from ship_simulation.optimizer.kemm_solver import ShipKEMMOptimizer
from ship_simulation.scenario.generator import ScenarioGenerator
from ship_simulation.visualization import ExperimentSeries, save_summary_dashboard


matplotlib.use("Agg")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")


class ShipSimulationSmokeTests(unittest.TestCase):
    def setUp(self):
        self.config = build_default_config()
        self.scenario = ScenarioGenerator(self.config).generate("crossing")
        self.interface = ShipOptimizerInterface(self.scenario, self.config)

    def test_optimizer_context_matches_problem_shape(self):
        context = self.interface.build_context()
        self.assertEqual(context.n_obj, 3)
        self.assertEqual(context.n_var, self.config.num_intermediate_waypoints * 3)

        obj_func = self.interface.make_objective_function(context)
        pop = np.vstack([context.initial_guess, context.initial_guess])
        values = obj_func(pop, 0.0)
        self.assertEqual(values.shape, (2, 3))

    def test_small_kemm_run_reaches_goal(self):
        demo = DemoConfig(
            optimizer_name="kemm",
            kemm=KEMMConfig(
                pop_size=16,
                generations=4,
                refresh_interval=2,
                seed=7,
                inject_initial_guess=True,
                initial_guess_copies=3,
                initial_guess_jitter_ratio=0.02,
            ),
        )
        result = ShipKEMMOptimizer(self.interface, demo).optimize()
        self.assertEqual(result.population.shape[1], self.interface.build_context().n_var)
        self.assertEqual(result.fitness.shape[1], 3)
        self.assertTrue(result.best_evaluation.reached_goal)

    def test_report_dashboard_is_generated(self):
        decision = self.interface.build_context().initial_guess
        evaluation = self.interface.simulate(decision)
        series = [ExperimentSeries(label="Initial", result=evaluation, color="tab:blue")]
        tmp_dir = Path("ship_simulation/outputs/test_artifacts")
        tmp_dir.mkdir(parents=True, exist_ok=True)
        try:
            output = tmp_dir / "dashboard.png"
            plot_config = ShipPlotConfig(
                style=PublicationStyle(dpi=180, title_size=12, label_size=10),
                dashboard_figsize=(12.0, 8.0),
            )
            save_summary_dashboard(output, self.scenario, series, plot_config=plot_config)
            self.assertTrue(output.exists())
            self.assertGreater(output.stat().st_size, 0)
        finally:
            if output.exists():
                try:
                    output.unlink()
                except PermissionError:
                    pass


if __name__ == "__main__":
    unittest.main()
