import json
import unittest
from pathlib import Path
from unittest.mock import patch

from apps.reporting import BenchmarkFigurePayload, BenchmarkPlotConfig, PublicationStyle, generate_all_figures
from kemm.core.types import KEMMChangeDiagnostics
from kemm.reporting import export_benchmark_report
from reporting_config import interactive_bundle_path


class _DummyConfig:
    POP_SIZE = 10
    N_VAR = 4
    N_OBJ = 2
    N_RUNS = 2
    SETTINGS = [(5, 10), (10, 10)]
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
        setting_results = {
            "5,10": results,
            "10,10": results,
        }
        ablation_setting_results = {
            "5,10": {
                "KEMM-Full": {
                    "FDA1": {"MIGD": [1.0, 1.1], "SP": [0.3, 0.4], "MS": [0.8, 0.82], "TIME": [0.1, 0.1]},
                    "FDA2": {"MIGD": [0.9, 1.0], "SP": [0.25, 0.3], "MS": [0.85, 0.86], "TIME": [0.1, 0.1]},
                }
            }
        }

        root = export_benchmark_report(
            results,
            _DummyConfig(),
            setting_results=setting_results,
            ablation_setting_results=ablation_setting_results,
        )

        self.assertTrue(root.exists())
        self.assertTrue((root / "raw" / "metrics.csv").exists())
        self.assertTrue((root / "raw" / "ranks.csv").exists())
        self.assertTrue((root / "raw" / "paper_table_metrics.csv").exists())
        self.assertTrue((root / "raw" / "ablation_setting_metrics.csv").exists())
        self.assertTrue((root / "raw" / "ablation_delta_metrics.csv").exists())
        self.assertTrue((root / "raw" / "summary.json").exists())
        self.assertTrue((root / "reports" / "summary.md").exists())

        payload = json.loads((root / "raw" / "summary.json").read_text(encoding="utf-8"))
        self.assertIn("metrics", payload)
        self.assertIn("ranks", payload)
        self.assertIn("paper_table_metrics", payload)
        self.assertIn("ablation_delta_metrics", payload)
        self.assertEqual(len(payload["metrics"]), 4)
        self.assertEqual(payload["ranks"][0]["algorithm"], "A")
        summary_md = (root / "reports" / "summary.md").read_text(encoding="utf-8")
        self.assertIn("Paper-Style MIGD Table", summary_md)
        self.assertIn("Ablation Delta Summary", summary_md)

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
            setting_results={
                "5,10": results,
                "10,10": results,
            },
            igd_curves=igd_curves,
            hv_curves={
                "KEMM": {"FDA1": [[0.2, 0.3, 0.35], [0.18, 0.29, 0.36]]},
                "KF": {"FDA1": [[0.1, 0.14, 0.18], [0.09, 0.13, 0.17]]},
            },
            diagnostics=diagnostics,
            ablation_results={
                "KEMM-Full": {
                    "FDA1": {"MIGD": [1.0, 1.05], "SP": [0.31, 0.33], "MS": [0.82, 0.83], "TIME": [0.1, 0.1]},
                },
                "MMTL-Original": {
                    "FDA1": {"MIGD": [1.25, 1.2], "SP": [0.42, 0.43], "MS": [0.72, 0.73], "TIME": [0.1, 0.1]},
                },
            },
            ablation_setting_results={
                "5,10": {
                    "KEMM-Full": {
                        "FDA1": {"MIGD": [1.0, 1.05], "SP": [0.31, 0.33], "MS": [0.82, 0.83], "TIME": [0.1, 0.1]},
                    },
                    "KEMM-NoMemory": {
                        "FDA1": {"MIGD": [1.12, 1.1], "SP": [0.36, 0.37], "MS": [0.79, 0.78], "TIME": [0.1, 0.1]},
                    },
                },
                "10,10": {
                    "KEMM-Full": {
                        "FDA1": {"MIGD": [0.95, 0.98], "SP": [0.28, 0.29], "MS": [0.84, 0.85], "TIME": [0.1, 0.1]},
                    },
                    "KEMM-NoMemory": {
                        "FDA1": {"MIGD": [1.08, 1.06], "SP": [0.35, 0.36], "MS": [0.8, 0.81], "TIME": [0.1, 0.1]},
                    },
                },
            },
            plot_config=BenchmarkPlotConfig(
                style=PublicationStyle(dpi=180, title_size=12, label_size=10),
                rank_bar_width=8.0,
                rank_bar_height=3.6,
                appendix_plots=True,
                interactive_figures=True,
            ),
        )

        generate_all_figures(payload=payload, output_prefix=str(output_dir / "benchmark"))

        self.assertTrue((output_dir / "benchmark_migd_table.png").exists())
        self.assertTrue((output_dir / "benchmark_migd_bar.png").exists())
        self.assertTrue((output_dir / "benchmark_metrics_grid.png").exists())
        self.assertTrue((output_dir / "benchmark_heatmap.png").exists())
        self.assertTrue((output_dir / "benchmark_rank_bar.png").exists())
        self.assertTrue((output_dir / "benchmark_igd_time.png").exists())
        self.assertTrue((output_dir / "benchmark_hv_time.png").exists())
        self.assertTrue((output_dir / "benchmark_operator_ratios.png").exists())
        self.assertTrue((output_dir / "benchmark_response_quality.png").exists())
        self.assertTrue((output_dir / "benchmark_prediction_confidence.png").exists())
        self.assertTrue((output_dir / "benchmark_change_dashboard.png").exists())
        self.assertTrue((output_dir / "benchmark_pairwise_wins.png").exists())
        self.assertTrue((output_dir / "benchmark_ablation.png").exists())
        self.assertTrue(interactive_bundle_path(output_dir / "benchmark_metrics_grid.png").exists())

    def test_generate_all_figures_skips_appendix_plot_by_default(self):
        results = {
            "KEMM": {"FDA1": {"MIGD": [1.0, 1.1], "SP": [0.3, 0.35], "MS": [0.8, 0.82], "TIME": [0.1, 0.1]}},
            "KF": {"FDA1": {"MIGD": [1.3, 1.4], "SP": [0.45, 0.5], "MS": [0.7, 0.72], "TIME": [0.1, 0.1]}},
        }
        output_dir = Path("benchmark_outputs") / "reporting_payload_no_appendix"
        payload = BenchmarkFigurePayload(
            results=results,
            problems=["FDA1"],
            plot_config=BenchmarkPlotConfig(style=PublicationStyle(dpi=120)),
        )
        generate_all_figures(payload=payload, output_prefix=str(output_dir / "benchmark"))
        self.assertTrue((output_dir / "benchmark_metrics_grid.png").exists())
        self.assertFalse((output_dir / "benchmark_migd_bar.png").exists())

    def test_generate_all_figures_falls_back_without_scienceplots(self):
        results = {
            "KEMM": {"FDA1": {"MIGD": [1.0, 1.1], "SP": [0.3, 0.35], "MS": [0.8, 0.82], "TIME": [0.1, 0.1]}},
            "KF": {"FDA1": {"MIGD": [1.3, 1.4], "SP": [0.45, 0.5], "MS": [0.7, 0.72], "TIME": [0.1, 0.1]}},
        }
        output_dir = Path("benchmark_outputs") / "reporting_payload_science_fallback"
        payload = BenchmarkFigurePayload(
            results=results,
            problems=["FDA1"],
            plot_config=BenchmarkPlotConfig(
                style=PublicationStyle(
                    dpi=120,
                    use_scienceplots=True,
                    science_styles=("science", "ieee", "no-latex"),
                )
            ),
        )
        with patch("reporting_config.HAS_SCIENCEPLOTS", False):
            generate_all_figures(payload=payload, output_prefix=str(output_dir / "benchmark"))
        self.assertTrue((output_dir / "benchmark_metrics_grid.png").exists())


if __name__ == "__main__":
    unittest.main()
