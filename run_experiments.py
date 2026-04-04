"""Thin compatibility wrapper for the benchmark application entrypoint."""

from apps.benchmark_runner import ExperimentConfig, ExperimentRunner, ResultPresenter, main, run_benchmark

__all__ = [
    "ExperimentConfig",
    "ExperimentRunner",
    "ResultPresenter",
    "main",
    "run_benchmark",
]


if __name__ == "__main__":
    main()
