import json
import unittest
from pathlib import Path

from apps.reporting import BenchmarkFigurePayload, BenchmarkPlotConfig, PublicationStyle, generate_all_figures
from kemm.core.types import KEMMChangeDiagnostics
from kemm.reporting import export_benchmark_report


class _DummyConfig:
    POP_SIZE = 10
    N_VAR = 4
    N_OBJ = 2
    N_RUNS = 2
    PROBLEMS = ["FDA1", "FDA2"]
    ALGORITHMS = {"A": object(), "B": object()}


class BenchmarkReportingTests(unittest.TestCase):
    def test_export_benchmark_report_creates_expected_files(self):
        results = {
            "A": {
                "FDA1": {"MIGD": [1.0, 1.2], "SP": [0.4, 0.5], "MS": [0.8, 0.9], "TIME": [0.1, 0.2]},
                "FDA2": {"MIGD": [0.9, 1.1], "SP": [0.3, 0.4], "MS": [0.7, 0.8], "TIME": [0.2, 0.2]},
            },
            "B": {
                "FDA1": {"MIGD": [1.5, 1.4], "SP": [0.6, 0.7], "MS": [0.6, 0.65], "TIME": [0.3, 0.4]},
                "FDA2": {"MIGD": [1.2, 1.3], "SP": [0.5, 0.55], "MS": [0.5, 0.6], "TIME": [0.3, 0.35]},
            },
        }

        root = export_benchmark_report(results, _DummyConfig())

        self.assertTrue(root.exists())
        self.assertTrue((root / "raw" / "metrics.csv").exists())
        self.assertTrue((root / "raw" / "ranks.csv").exists())
        self.assertTrue((root / "raw" / "summary.json").exists())
        self.assertTrue((root / "reports" / "summary.md").exists())

        payload = json.loads((root / "raw" / "summary.json").read_text(encoding="utf-8"))
        self.assertIn("metrics", payload)
        self.assertIn("ranks", payload)
        self.assertEqual(len(payload["metrics"]), 4)
        self.assertEqual(payload["ranks"][0]["algorithm"], "A")

    def test_generate_all_figures_accepts_structured_payload(self):
        results = {
            "KEMM": {
                "FDA1": {"MIGD": [1.0, 1.1], "SP": [0.3, 0.35], "MS": [0.8, 0.82], "TIME": [0.1, 0.1]},
            },
            "KF": {
                "FDA1": {"MIGD": [1.3, 1.4], "SP": [0.45, 0.5], "MS": [0.7, 0.72], "TIME": [0.1, 0.1]},
            },
        }
        igd_curves = {
            "KEMM": {"FDA1": [[1.5, 1.2, 1.0], [1.4, 1.15, 1.05]]},
            "KF": {"FDA1": [[1.8, 1.6, 1.4], [1.75, 1.55, 1.35]]},
        }
        diagnostics = {
            "KEMM": {
                "FDA1": [[
                    KEMMChangeDiagnostics(
                        time_step=1,
                        change_time=0.1,
                        change_magnitude=0.2,
                        prediction_confidence=0.75,
                        transferability=0.6,
                        operator_ratios={
                            "memory": 0.25,
                            "prediction": 0.35,
                            "transfer": 0.25,
                            "reinit": 0.15,
                        },
                        requested_counts={
                            "memory": 10,
                            "prediction": 12,
                            "transfer": 8,
                            "reinit": 5,
                            "prior": 0,
                        },
                        actual_counts={
                            "memory": 10,
                            "prediction": 11,
                            "transfer": 8,
                            "reinit": 5,
                            "prior": 0,
                            "elite": 4,
                            "previous": 6,
                        },
                        candidate_pool_size=44,
                        response_quality=0.68,
                        selected_front_size=10,
                    ),
                    KEMMChangeDiagnostics(
                        time_step=2,
                        change_time=0.2,
                        change_magnitude=0.35,
                        prediction_confidence=0.62,
                        transferability=0.55,
                        operator_ratios={
                            "memory": 0.2,
                            "prediction": 0.4,
                            "transfer": 0.25,
                            "reinit": 0.15,
                        },
                        requested_counts={
                            "memory": 9,
                            "prediction": 13,
                            "transfer": 8,
                            "reinit": 6,
                            "prior": 0,
                        },
                        actual_counts={
                            "memory": 9,
                            "prediction": 12,
                            "transfer": 8,
                            "reinit": 6,
                            "prior": 0,
                            "elite": 4,
                            "previous": 5,
                        },
                        candidate_pool_size=44,
                        response_quality=0.72,
                        selected_front_size=10,
                    ),
                ]]
            }
        }

        output_dir = Path("benchmark_outputs") / "reporting_payload_test"
        payload = BenchmarkFigurePayload(
            results=results,
            problems=["FDA1"],
            igd_curves=igd_curves,
            diagnostics=diagnostics,
            plot_config=BenchmarkPlotConfig(
                style=PublicationStyle(dpi=180, title_size=12, label_size=10),
                rank_bar_width=8.0,
                rank_bar_height=3.6,
            ),
        )

        generate_all_figures(payload=payload, output_prefix=str(output_dir / "benchmark"))

        self.assertTrue((output_dir / "benchmark_migd_bar.png").exists())
        self.assertTrue((output_dir / "benchmark_metrics_grid.png").exists())
        self.assertTrue((output_dir / "benchmark_heatmap.png").exists())
        self.assertTrue((output_dir / "benchmark_rank_bar.png").exists())
        self.assertTrue((output_dir / "benchmark_igd_time.png").exists())
        self.assertTrue((output_dir / "benchmark_operator_ratios.png").exists())
        self.assertTrue((output_dir / "benchmark_response_quality.png").exists())
        self.assertTrue((output_dir / "benchmark_prediction_confidence.png").exists())
        self.assertTrue((output_dir / "benchmark_change_dashboard.png").exists())
        self.assertTrue((output_dir / "benchmark_pairwise_wins.png").exists())


if __name__ == "__main__":
    unittest.main()
