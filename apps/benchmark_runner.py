"""Benchmark application runner.

这个文件是 benchmark 主线的真实应用入口。

职责分成三类：

1. 定义实验配置
2. 批量执行“算法 × 问题 × 重复运行”实验
3. 把结果整理成控制台表格、图表和结构化报告

说明：

- 根目录 `run_experiments.py` 现在只是 thin wrapper
- benchmark 主线的真实逻辑以本文件为准
- 这里仍然同时包含 runner 和 presenter，后续还可以继续拆分
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass, replace
import hashlib
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from kemm.adapters import BenchmarkPriorAdapter
from kemm.algorithms import (
    KF_DMOEA,
    KEMM_DMOEA_Improved,
    MMTL_DMOEA,
    PPS_DMOEA,
    RI_DMOEA,
    SVR_DMOEA,
    Tr_DMOEA,
)
from kemm.benchmark import DynamicTestProblems, PerformanceMetrics
from kemm.core.types import ExperimentConfig as SharedExperimentConfig
from kemm.core.types import KEMMChangeDiagnostics, KEMMConfig as RuntimeKEMMConfig
from kemm.reporting import build_report_paths, export_benchmark_report
from reporting_config import build_benchmark_plot_config

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MPL = True
except ImportError:
    HAS_MPL = False

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")


_SHARED_EXPERIMENT_DEFAULTS = SharedExperimentConfig()
CACHE_SCHEMA_VERSION = "benchmark-task-cache-v1"
_BENCHMARK_TASK_CACHE_DIR = Path("benchmark_outputs") / "_cache" / "benchmark_tasks"


@contextmanager
def _temporary_numpy_seed(seed: int):
    caller_state = np.random.get_state()
    np.random.seed(int(seed))
    try:
        yield
    finally:
        np.random.set_state(caller_state)


def _recommended_worker_count(limit: int = 4) -> int:
    cpu_count = os.cpu_count() or 1
    if cpu_count <= 1:
        return 1
    return max(1, min(limit, cpu_count - 1))


def _benchmark_config_snapshot(cfg: "ExperimentConfig") -> dict[str, object]:
    return {
        "POP_SIZE": int(getattr(cfg, "POP_SIZE")),
        "N_VAR": int(getattr(cfg, "N_VAR")),
        "N_OBJ": int(getattr(cfg, "N_OBJ")),
        "N_CHANGES": int(getattr(cfg, "N_CHANGES")),
        "GENS_PER_CHANGE": int(getattr(cfg, "GENS_PER_CHANGE")),
        "KEMM_CONFIG": replace(getattr(cfg, "KEMM_CONFIG", RuntimeKEMMConfig())) if getattr(cfg, "KEMM_CONFIG", None) is not None else RuntimeKEMMConfig(),
    }


def _json_ready(value):
    if is_dataclass(value):
        return _json_ready(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_ready(val) for key, val in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return np.asarray(value).tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, type):
        return f"{value.__module__}.{value.__qualname__}"
    return value


def _problem_aux_snapshot(problem_suite: DynamicTestProblems, pof_func, *, n_changes: int, gens_per_change: int) -> dict[str, object]:
    generation = 0
    times: list[float] = []
    true_pof_series: list[list[list[float]]] = []
    for _change_index in range(int(n_changes)):
        t = float(problem_suite.get_time(generation))
        times.append(t)
        try:
            true_pof = pof_func(t=t)
        except TypeError:
            true_pof = pof_func()
        true_pof_series.append(np.asarray(true_pof, dtype=float).tolist())
        generation += int(gens_per_change)
    return {"times": times, "true_pof_series": true_pof_series}


def _ensure_problem_aux(
    resource: dict[str, object],
    *,
    n_changes: int,
    gens_per_change: int,
) -> dict[str, object]:
    problem_aux = resource.get("problem_aux")
    if problem_aux is None:
        problem_aux = _problem_aux_snapshot(
            resource["problem_suite"],
            resource["pof_func"],
            n_changes=n_changes,
            gens_per_change=gens_per_change,
        )
        resource["problem_aux"] = problem_aux
    return dict(problem_aux)


def _benchmark_algo_snapshot(algo_name: str, algo_spec) -> dict[str, object]:
    if isinstance(algo_spec, dict):
        return {
            "algo_name": str(algo_name),
            "algorithm": _json_ready(algo_spec.get("algorithm", KEMM_DMOEA_Improved)),
            "config_overrides": _json_ready(dict(algo_spec.get("config_overrides", {}))),
            "benchmark_aware_prior": bool(algo_spec.get("benchmark_aware_prior", True)),
            "ablation_variant": str(algo_name) if str(algo_name).startswith("KEMM-") else None,
        }
    return {
        "algo_name": str(algo_name),
        "algorithm": _json_ready(algo_spec),
        "config_overrides": {},
        "benchmark_aware_prior": True,
        "ablation_variant": None,
    }


def _benchmark_task_cache_key(task: dict[str, object]) -> str:
    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "cfg_snapshot": _json_ready(task["cfg_snapshot"]),
        "algo_spec": _benchmark_algo_snapshot(str(task["algo_name"]), task["algo_spec"]),
        "problem": str(task["prob_name"]),
        "run": int(task["run"]),
        "setting": str(task["setting_key"]),
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _benchmark_task_cache_path(task: dict[str, object]) -> Path:
    root = Path(str(task.get("cache_dir") or _BENCHMARK_TASK_CACHE_DIR))
    key = _benchmark_task_cache_key(task)
    return root / key[:2] / f"{key}.json"


def _serialize_change_diagnostics(history) -> list[dict[str, object]]:
    payload: list[dict[str, object]] = []
    for item in history or []:
        if isinstance(item, KEMMChangeDiagnostics):
            payload.append(_json_ready(asdict(item)))
        elif isinstance(item, dict):
            payload.append(_json_ready(item))
    return payload


def _restore_change_diagnostics(history) -> list[KEMMChangeDiagnostics]:
    restored: list[KEMMChangeDiagnostics] = []
    for item in history or []:
        if isinstance(item, KEMMChangeDiagnostics):
            restored.append(item)
        elif isinstance(item, dict):
            restored.append(KEMMChangeDiagnostics(**item))
    return restored


def _write_benchmark_task_cache(path: Path, result: dict[str, object]) -> None:
    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "result": {
            "setting_key": str(result["setting_key"]),
            "algo_name": str(result["algo_name"]),
            "prob_name": str(result["prob_name"]),
            "run": int(result["run"]),
            "migd": float(result["migd"]),
            "sp": float(result["sp"]),
            "ms": float(result["ms"]),
            "time": float(result["time"]),
            "igd_curve": [float(value) for value in result["igd_curve"]],
            "hv_curve": [float(value) for value in result["hv_curve"]],
            "change_diagnostics": _serialize_change_diagnostics(result["change_diagnostics"]),
            "nt": int(result["nt"]),
            "tau_t": int(result["tau_t"]),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _load_benchmark_task_cache(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if str(payload.get("schema_version")) != CACHE_SCHEMA_VERSION:
        return None
    result = dict(payload.get("result") or {})
    if not result:
        return None
    result["migd"] = float(result["migd"])
    result["sp"] = float(result["sp"])
    result["ms"] = float(result["ms"])
    result["time"] = float(result["time"])
    result["run"] = int(result["run"])
    result["nt"] = int(result["nt"])
    result["tau_t"] = int(result["tau_t"])
    result["igd_curve"] = [float(value) for value in result.get("igd_curve", [])]
    result["hv_curve"] = [float(value) for value in result.get("hv_curve", [])]
    result["change_diagnostics"] = _restore_change_diagnostics(result.get("change_diagnostics", []))
    return result


def _instantiate_algorithm_from_snapshot(cfg_snapshot: dict[str, object], algo_spec, lb: np.ndarray, ub: np.ndarray):
    algo_class = algo_spec
    config_overrides = {}
    benchmark_aware_prior = True
    if isinstance(algo_spec, dict):
        algo_class = algo_spec.get("algorithm", KEMM_DMOEA_Improved)
        config_overrides = dict(algo_spec.get("config_overrides", {}))
        benchmark_aware_prior = bool(algo_spec.get("benchmark_aware_prior", True))

    if algo_class is KEMM_DMOEA_Improved:
        base_config = cfg_snapshot["KEMM_CONFIG"] or RuntimeKEMMConfig()
        kemm_config = replace(
            base_config,
            pop_size=int(cfg_snapshot["POP_SIZE"]),
            n_var=int(cfg_snapshot["N_VAR"]),
            n_obj=int(cfg_snapshot["N_OBJ"]),
            benchmark_aware_prior=benchmark_aware_prior,
            **config_overrides,
        )
        return algo_class(
            int(cfg_snapshot["POP_SIZE"]),
            int(cfg_snapshot["N_VAR"]),
            int(cfg_snapshot["N_OBJ"]),
            (lb, ub),
            config=kemm_config,
            benchmark_adapter=BenchmarkPriorAdapter() if benchmark_aware_prior else None,
            benchmark_aware_prior=benchmark_aware_prior,
        )
    return algo_class(int(cfg_snapshot["POP_SIZE"]), int(cfg_snapshot["N_VAR"]), int(cfg_snapshot["N_OBJ"]), (lb, ub))


def _run_benchmark_case(task: dict[str, object]) -> dict[str, object]:
    cache_enabled = bool(task.get("cache_enabled", False))
    force_rerun = bool(task.get("force_rerun", False))
    cache_path = _benchmark_task_cache_path(task)
    if cache_enabled and not force_rerun:
        cached = _load_benchmark_task_cache(cache_path)
        if cached is not None:
            cached["cache_hit"] = True
            return cached

    cfg_snapshot = dict(task["cfg_snapshot"])
    nt = int(task["nt"])
    tau_t = int(task["tau_t"])
    problem_suite = DynamicTestProblems(nt=nt, tau_t=tau_t)
    obj_func, pof_func = problem_suite.get_problem(str(task["prob_name"]))
    problem_aux = dict(task.get("problem_aux") or {})
    if not problem_aux:
        problem_aux = _problem_aux_snapshot(
            problem_suite,
            pof_func,
            n_changes=int(cfg_snapshot["N_CHANGES"]),
            gens_per_change=int(cfg_snapshot["GENS_PER_CHANGE"]),
        )
    precomputed_times = [float(value) for value in problem_aux.get("times", [])]
    precomputed_true_pofs = [np.asarray(points, dtype=float) for points in problem_aux.get("true_pof_series", [])]

    n_var = int(cfg_snapshot["N_VAR"])
    lb = np.zeros(n_var)
    ub = np.ones(n_var)
    lb[1:] = -1.0
    ub[1:] = 1.0

    algo = _instantiate_algorithm_from_snapshot(cfg_snapshot, task["algo_spec"], lb, ub)
    with _temporary_numpy_seed(int(task["run_seed"])):
        algo.initialize()
        t0 = time.time()

        igd_list: List[float] = []
        hv_list: List[float] = []
        sp_list: List[float] = []
        ms_list: List[float] = []
        generation = 0
        metrics = PerformanceMetrics()

        for change_index in range(int(cfg_snapshot["N_CHANGES"])):
            if change_index < len(precomputed_times):
                t = precomputed_times[change_index]
            else:
                t = problem_suite.get_time(generation)

            if change_index == 0:
                algo.fitness = algo.evaluate(algo.population, obj_func, t)
            else:
                algo.respond_to_change(obj_func, t)

            for _ in range(int(cfg_snapshot["GENS_PER_CHANGE"])):
                algo.evolve_one_gen(obj_func, t)

            generation += int(cfg_snapshot["GENS_PER_CHANGE"])
            obtained = algo.get_pareto_front()

            if change_index < len(precomputed_true_pofs):
                true_pof = precomputed_true_pofs[change_index]
            else:
                try:
                    true_pof = pof_func(t=t)
                except TypeError:
                    true_pof = pof_func()
            ref_point = np.max(np.vstack([true_pof, obtained]), axis=0) + 0.1

            igd_list.append(metrics.igd(obtained, true_pof))
            hv_list.append(metrics.hypervolume(obtained, ref_point))
            sp_list.append(metrics.spacing(obtained))
            ms_list.append(metrics.maximum_spread(obtained, true_pof))

    result = {
        "setting_key": str(task["setting_key"]),
        "algo_name": str(task["algo_name"]),
        "prob_name": str(task["prob_name"]),
        "run": int(task["run"]),
        "migd": metrics.migd(igd_list),
        "sp": float(np.mean(sp_list)),
        "ms": float(np.mean(ms_list)),
        "time": time.time() - t0,
        "igd_curve": igd_list,
        "hv_curve": hv_list,
        "change_diagnostics": list(getattr(algo, "change_diagnostics_history", [])),
        "nt": nt,
        "tau_t": tau_t,
    }
    if cache_enabled:
        _write_benchmark_task_cache(cache_path, result)
    result["cache_hit"] = False
    return result


class ExperimentConfig:
    """Benchmark 实验配置。

    这个类把 benchmark 主线里最常改的参数集中管理，包括：

    - 问题规模参数
    - 动态环境变化参数
    - 重复运行次数
    - 算法集合
    - 消融算法集合

    这样做的目的是让 quick/full 模式切换和论文复现实验更可控。
    """

    POP_SIZE = _SHARED_EXPERIMENT_DEFAULTS.pop_size
    N_VAR = _SHARED_EXPERIMENT_DEFAULTS.n_var
    N_OBJ = _SHARED_EXPERIMENT_DEFAULTS.n_obj
    NT = _SHARED_EXPERIMENT_DEFAULTS.nt
    TAU_T = _SHARED_EXPERIMENT_DEFAULTS.tau_t
    SETTINGS = [(5, 10), (10, 10), (10, 20)]
    CANONICAL_SETTING = (10, 10)
    N_CHANGES = _SHARED_EXPERIMENT_DEFAULTS.n_changes
    GENS_PER_CHANGE = _SHARED_EXPERIMENT_DEFAULTS.gens_per_change
    N_RUNS = _SHARED_EXPERIMENT_DEFAULTS.n_runs
    SIGNIFICANCE = _SHARED_EXPERIMENT_DEFAULTS.significance

    PROBLEMS_STANDARD = list(_SHARED_EXPERIMENT_DEFAULTS.problems_standard)
    PROBLEMS_JY = list(_SHARED_EXPERIMENT_DEFAULTS.problems_jy)
    PROBLEMS = PROBLEMS_STANDARD

    ALGORITHMS = {
        "RI": RI_DMOEA,
        "PPS": PPS_DMOEA,
        "KF": KF_DMOEA,
        "SVR": SVR_DMOEA,
        "Tr": Tr_DMOEA,
        "MMTL": MMTL_DMOEA,
        "KEMM": KEMM_DMOEA_Improved,
    }

    ABLATION_VARIANTS = {
        "KEMM-Full": {"config_overrides": {}},
        "KEMM-NoMemory": {"config_overrides": {"enable_memory": False}},
        "KEMM-NoPrediction": {"config_overrides": {"enable_prediction": False}},
        "KEMM-NoTransfer": {"config_overrides": {"enable_transfer": False}},
        "KEMM-NoAdaptive": {"config_overrides": {"enable_adaptive": False}},
    }
    ABLATION_BENCHMARK_PRIOR = False
    KEMM_CONFIG = RuntimeKEMMConfig()
    MAX_WORKERS = 1
    CACHE_ENABLED = False
    FORCE_RERUN = False
    CACHE_DIR = str(_BENCHMARK_TASK_CACHE_DIR)

    def __init__(self):
        self.POP_SIZE = int(self.__class__.POP_SIZE)
        self.N_VAR = int(self.__class__.N_VAR)
        self.N_OBJ = int(self.__class__.N_OBJ)
        self.NT = int(self.__class__.NT)
        self.TAU_T = int(self.__class__.TAU_T)
        self.SETTINGS = [tuple(map(int, setting)) for setting in self.__class__.SETTINGS]
        self.CANONICAL_SETTING = tuple(map(int, self.__class__.CANONICAL_SETTING))
        self.N_CHANGES = int(self.__class__.N_CHANGES)
        self.GENS_PER_CHANGE = int(self.__class__.GENS_PER_CHANGE)
        self.N_RUNS = int(self.__class__.N_RUNS)
        self.SIGNIFICANCE = float(self.__class__.SIGNIFICANCE)
        self.PROBLEMS_STANDARD = list(self.__class__.PROBLEMS_STANDARD)
        self.PROBLEMS_JY = list(self.__class__.PROBLEMS_JY)
        self.PROBLEMS = list(self.__class__.PROBLEMS)
        self.ALGORITHMS = dict(self.__class__.ALGORITHMS)
        self.ABLATION_VARIANTS = {
            name: {
                key: (dict(value) if isinstance(value, dict) else value)
                for key, value in spec.items()
            }
            for name, spec in self.__class__.ABLATION_VARIANTS.items()
        }
        self.ABLATION_BENCHMARK_PRIOR = bool(self.__class__.ABLATION_BENCHMARK_PRIOR)
        self.KEMM_CONFIG = replace(self.__class__.KEMM_CONFIG)
        self.MAX_WORKERS = int(self.__class__.MAX_WORKERS)
        self.CACHE_ENABLED = bool(self.__class__.CACHE_ENABLED)
        self.FORCE_RERUN = bool(self.__class__.FORCE_RERUN)
        self.CACHE_DIR = str(self.__class__.CACHE_DIR)


class ExperimentRunner:
    """benchmark 批量实验运行器。"""

    def __init__(self, config: ExperimentConfig | None = None):
        # `results` 保存最终统计量。
        # `igd_curves` 保存环境变化过程中的 IGD 曲线，后续画过程图会用到。
        # `algorithm_diagnostics` 保存算法变化响应的结构化诊断，供可视化和调试使用。
        self.cfg = config or ExperimentConfig()
        self.problems = DynamicTestProblems(nt=self.cfg.NT, tau_t=self.cfg.TAU_T)
        self.metrics = PerformanceMetrics()
        self.setting_results: Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]] = {}
        self.setting_igd_curves: Dict[str, Dict[str, Dict[str, List[List[float]]]]] = {}
        self.setting_hv_curves: Dict[str, Dict[str, Dict[str, List[List[float]]]]] = {}
        self.setting_algorithm_diagnostics: Dict[str, Dict[str, Dict[str, List[list]]]] = {}
        self.results: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
        self.igd_curves: Dict[str, Dict[str, List[List[float]]]] = {}
        self.hv_curves: Dict[str, Dict[str, List[List[float]]]] = {}
        self.algorithm_diagnostics: Dict[str, Dict[str, List[list]]] = {}
        self.ablation_setting_results: Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]] = {}
        self.ablation_results: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
        self.task_cache_hits = 0
        self.task_cache_misses = 0

    @staticmethod
    def _setting_key(setting: tuple[int, int]) -> str:
        return f"{int(setting[0])},{int(setting[1])}"

    @staticmethod
    def _parse_setting_key(setting_key: str) -> tuple[int, int]:
        nt_str, tau_t_str = setting_key.split(",", maxsplit=1)
        return int(nt_str), int(tau_t_str)

    def _canonical_setting_key(self) -> str:
        return self._setting_key(tuple(self.cfg.CANONICAL_SETTING))

    def _instantiate_algorithm(self, algo_spec, lb: np.ndarray, ub: np.ndarray):
        algo_class = algo_spec
        config_overrides = {}
        benchmark_aware_prior = True
        if isinstance(algo_spec, dict):
            algo_class = algo_spec.get("algorithm", KEMM_DMOEA_Improved)
            config_overrides = dict(algo_spec.get("config_overrides", {}))
            benchmark_aware_prior = bool(algo_spec.get("benchmark_aware_prior", True))

        if algo_class is KEMM_DMOEA_Improved:
            base_config = self.cfg.KEMM_CONFIG or RuntimeKEMMConfig()
            kemm_config = replace(
                base_config,
                pop_size=self.cfg.POP_SIZE,
                n_var=self.cfg.N_VAR,
                n_obj=self.cfg.N_OBJ,
                benchmark_aware_prior=benchmark_aware_prior,
                **config_overrides,
            )
            return algo_class(
                self.cfg.POP_SIZE,
                self.cfg.N_VAR,
                self.cfg.N_OBJ,
                (lb, ub),
                config=kemm_config,
                benchmark_adapter=BenchmarkPriorAdapter() if benchmark_aware_prior else None,
                benchmark_aware_prior=benchmark_aware_prior,
            )
        return algo_class(self.cfg.POP_SIZE, self.cfg.N_VAR, self.cfg.N_OBJ, (lb, ub))

    def _initialize_metric_bucket(self, algorithms: Dict[str, object]) -> Dict[str, Dict[str, Dict[str, List[float]]]]:
        return {
            algo_name: {
                prob_name: {"MIGD": [], "SP": [], "MS": [], "TIME": []}
                for prob_name in self.cfg.PROBLEMS
            }
            for algo_name in algorithms
        }

    def _initialize_curve_bucket(self, algorithms: Dict[str, object]) -> Dict[str, Dict[str, List[List[float]]]]:
        return {
            algo_name: {prob_name: [] for prob_name in self.cfg.PROBLEMS}
            for algo_name in algorithms
        }

    @staticmethod
    def _store_setting_outcome(
        *,
        setting_results,
        setting_igd_curves,
        setting_hv_curves,
        setting_diagnostics,
        result: dict[str, object],
        collect_curves: bool,
        collect_diagnostics: bool,
    ) -> None:
        setting_key = str(result["setting_key"])
        algo_name = str(result["algo_name"])
        prob_name = str(result["prob_name"])
        metrics = setting_results[setting_key][algo_name][prob_name]
        metrics["MIGD"].append(float(result["migd"]))
        metrics["SP"].append(float(result["sp"]))
        metrics["MS"].append(float(result["ms"]))
        metrics["TIME"].append(float(result["time"]))
        if collect_curves:
            setting_igd_curves[setting_key][algo_name][prob_name].append(list(result["igd_curve"]))
            setting_hv_curves[setting_key][algo_name][prob_name].append(list(result["hv_curve"]))
        if collect_diagnostics:
            setting_diagnostics[setting_key][algo_name][prob_name].append(list(result["change_diagnostics"]))

    def _cache_enabled(self) -> bool:
        return bool(getattr(self.cfg, "CACHE_ENABLED", False))

    def _force_rerun(self) -> bool:
        return bool(getattr(self.cfg, "FORCE_RERUN", False))

    def _cache_dir(self) -> str:
        return str(getattr(self.cfg, "CACHE_DIR", _BENCHMARK_TASK_CACHE_DIR))

    def _run_cached_serial_task(self, task: dict[str, object], resource: dict[str, object]) -> dict[str, object]:
        cache_enabled = self._cache_enabled()
        force_rerun = self._force_rerun()
        cache_path = _benchmark_task_cache_path(task)
        if cache_enabled and not force_rerun:
            cached = _load_benchmark_task_cache(cache_path)
            if cached is not None:
                cached["cache_hit"] = True
                return cached

        problem_suite = resource["problem_suite"]
        obj_func = resource["obj_func"]
        pof_func = resource["pof_func"]
        with _temporary_numpy_seed(int(task["run_seed"])):
            result = self._run_single(
                task["algo_spec"],
                obj_func,
                pof_func,
                problem_suite=problem_suite,
                precomputed_aux=_ensure_problem_aux(
                    resource,
                    n_changes=self.cfg.N_CHANGES,
                    gens_per_change=self.cfg.GENS_PER_CHANGE,
                ),
            )
        serial_result = {
            "setting_key": str(task["setting_key"]),
            "algo_name": str(task["algo_name"]),
            "prob_name": str(task["prob_name"]),
            "run": int(task["run"]),
            "migd": float(result["migd"]),
            "sp": float(result["sp"]),
            "ms": float(result["ms"]),
            "time": float(result["time"]),
            "igd_curve": list(result["igd_curve"]),
            "hv_curve": list(result["hv_curve"]),
            "change_diagnostics": list(result["change_diagnostics"]),
            "nt": int(task["nt"]),
            "tau_t": int(task["tau_t"]),
            "cache_hit": False,
        }
        if cache_enabled:
            _write_benchmark_task_cache(cache_path, serial_result)
        return serial_result

    def _run_setting_sweep(
        self,
        algorithms: Dict[str, object],
        *,
        progress_prefix: str,
        collect_curves: bool,
        collect_diagnostics: bool,
    ):
        setting_results: Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]] = {}
        setting_igd_curves: Dict[str, Dict[str, Dict[str, List[List[float]]]]] = {}
        setting_hv_curves: Dict[str, Dict[str, Dict[str, List[List[float]]]]] = {}
        setting_diagnostics: Dict[str, Dict[str, Dict[str, List[list]]]] = {}
        configured_workers = max(1, int(getattr(self.cfg, "MAX_WORKERS", 1)))
        tasks: list[dict[str, object]] = []
        problem_resources: dict[tuple[str, str], dict[str, object]] = {}
        tasks_by_problem: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
        needs_problem_aux: set[tuple[str, str]] = set()
        cfg_snapshot = _benchmark_config_snapshot(self.cfg)
        canonical_key = self._canonical_setting_key()
        self.task_cache_hits = 0
        self.task_cache_misses = 0
        for setting in self.cfg.SETTINGS:
            nt, tau_t = int(setting[0]), int(setting[1])
            setting_key = self._setting_key((nt, tau_t))
            setting_results[setting_key] = self._initialize_metric_bucket(algorithms)
            collect_curves_for_setting = collect_curves and setting_key == canonical_key
            collect_diagnostics_for_setting = collect_diagnostics and setting_key == canonical_key
            if collect_curves_for_setting:
                setting_igd_curves[setting_key] = self._initialize_curve_bucket(algorithms)
                setting_hv_curves[setting_key] = self._initialize_curve_bucket(algorithms)
            if collect_diagnostics_for_setting:
                setting_diagnostics[setting_key] = self._initialize_curve_bucket(algorithms)

            problem_suite = DynamicTestProblems(nt=nt, tau_t=tau_t)
            for prob_name in self.cfg.PROBLEMS:
                obj_func, pof_func = problem_suite.get_problem(prob_name)
                problem_resources[(setting_key, prob_name)] = {
                    "problem_suite": problem_suite,
                    "obj_func": obj_func,
                    "pof_func": pof_func,
                    "problem_aux": None,
                }

            for algo_name, algo_spec in algorithms.items():
                for prob_name in self.cfg.PROBLEMS:
                    for run in range(self.cfg.N_RUNS):
                        seed_offset = sum((index + 1) * ord(ch) for index, ch in enumerate(f"{setting_key}:{algo_name}")) % 10000
                        tasks.append(
                            {
                                "cfg_snapshot": cfg_snapshot,
                                "setting_key": setting_key,
                                "nt": nt,
                                "tau_t": tau_t,
                                "algo_name": algo_name,
                                "algo_spec": algo_spec,
                                "prob_name": prob_name,
                                "run": run,
                                "run_seed": run * 1000 + seed_offset,
                                "cache_enabled": self._cache_enabled(),
                                "force_rerun": self._force_rerun(),
                                "cache_dir": self._cache_dir(),
                            }
                        )
                        tasks_by_problem[(setting_key, prob_name)].append(tasks[-1])
                        if not self._cache_enabled() or self._force_rerun():
                            needs_problem_aux.add((setting_key, prob_name))
                        else:
                            cache_path = _benchmark_task_cache_path(tasks[-1])
                            if _load_benchmark_task_cache(cache_path) is None:
                                needs_problem_aux.add((setting_key, prob_name))

        for resource_key in needs_problem_aux:
            resource = problem_resources[resource_key]
            problem_aux = _ensure_problem_aux(
                resource,
                n_changes=self.cfg.N_CHANGES,
                gens_per_change=self.cfg.GENS_PER_CHANGE,
            )
            for task in tasks_by_problem[resource_key]:
                task["problem_aux"] = problem_aux

        total = len(tasks)
        counter = 0
        t_start = time.time()
        actual_workers = max(1, min(configured_workers, total))
        if actual_workers <= 1:
            for task in tasks:
                resource = problem_resources[(str(task["setting_key"]), str(task["prob_name"]))]
                result = self._run_cached_serial_task(task, resource)
                counter += 1
                if bool(result.get("cache_hit")):
                    self.task_cache_hits += 1
                else:
                    self.task_cache_misses += 1
                self._store_setting_outcome(
                    setting_results=setting_results,
                    setting_igd_curves=setting_igd_curves,
                    setting_hv_curves=setting_hv_curves,
                    setting_diagnostics=setting_diagnostics,
                    result=result,
                    collect_curves=collect_curves and str(result["setting_key"]) == canonical_key,
                    collect_diagnostics=collect_diagnostics and str(result["setting_key"]) == canonical_key,
                )
                elapsed = time.time() - t_start
                rate = counter / (elapsed + 1e-6)
                eta = (total - counter) / (rate + 1e-6)
                cache_label = "hit" if bool(result.get("cache_hit")) else "run"
                print(
                    f"\r  [{progress_prefix} {counter:4d}/{total}] ({int(result['nt']):>2d},{int(result['tau_t']):>2d}) "
                    f"{str(result['algo_name']):>16s}|{str(result['prob_name']):>5s}|R{int(result['run'])+1} | {cache_label:>3s} | "
                    f"{elapsed:.0f}s elapsed, ~{eta:.0f}s left",
                    end="",
                    flush=True,
                )
        else:
            print(f"  使用 {actual_workers} 个进程并行执行 {total} 个 benchmark 任务...", flush=True)
            try:
                with ProcessPoolExecutor(max_workers=actual_workers) as executor:
                    future_map = {executor.submit(_run_benchmark_case, task): task for task in tasks}
                    for future in as_completed(future_map):
                        result = future.result()
                        counter += 1
                        if bool(result.get("cache_hit")):
                            self.task_cache_hits += 1
                        else:
                            self.task_cache_misses += 1
                        self._store_setting_outcome(
                            setting_results=setting_results,
                            setting_igd_curves=setting_igd_curves,
                            setting_hv_curves=setting_hv_curves,
                            setting_diagnostics=setting_diagnostics,
                            result=result,
                            collect_curves=collect_curves and str(result["setting_key"]) == canonical_key,
                            collect_diagnostics=collect_diagnostics and str(result["setting_key"]) == canonical_key,
                        )
                        elapsed = time.time() - t_start
                        rate = counter / (elapsed + 1e-6)
                        eta = (total - counter) / (rate + 1e-6)
                        cache_label = "hit" if bool(result.get("cache_hit")) else "run"
                        print(
                            f"\r  [{progress_prefix} {counter:4d}/{total}] ({int(result['nt']):>2d},{int(result['tau_t']):>2d}) "
                            f"{str(result['algo_name']):>16s}|{str(result['prob_name']):>5s}|R{int(result['run'])+1} | {cache_label:>3s} | "
                            f"{elapsed:.0f}s elapsed, ~{eta:.0f}s left",
                            end="",
                            flush=True,
                        )
            except (PermissionError, OSError) as exc:
                print(f"\n  [WARN] 并行进程池不可用，回退到串行执行: {exc}", flush=True)
                counter = 0
                t_start = time.time()
                for task in tasks:
                    resource = problem_resources[(str(task["setting_key"]), str(task["prob_name"]))]
                    result = self._run_cached_serial_task(task, resource)
                    counter += 1
                    if bool(result.get("cache_hit")):
                        self.task_cache_hits += 1
                    else:
                        self.task_cache_misses += 1
                    self._store_setting_outcome(
                        setting_results=setting_results,
                        setting_igd_curves=setting_igd_curves,
                        setting_hv_curves=setting_hv_curves,
                        setting_diagnostics=setting_diagnostics,
                        result=result,
                        collect_curves=collect_curves and str(result["setting_key"]) == canonical_key,
                        collect_diagnostics=collect_diagnostics and str(result["setting_key"]) == canonical_key,
                    )
                    elapsed = time.time() - t_start
                    rate = counter / (elapsed + 1e-6)
                    eta = (total - counter) / (rate + 1e-6)
                    cache_label = "hit" if bool(result.get("cache_hit")) else "run"
                    print(
                        f"\r  [{progress_prefix} {counter:4d}/{total}] ({int(result['nt']):>2d},{int(result['tau_t']):>2d}) "
                        f"{str(result['algo_name']):>16s}|{str(result['prob_name']):>5s}|R{int(result['run'])+1} | {cache_label:>3s} | "
                        f"{elapsed:.0f}s elapsed, ~{eta:.0f}s left",
                        end="",
                        flush=True,
                    )
        print(
            f"\n  完成, 总耗时 {time.time() - t_start:.1f}s"
            + (
                f" | cache hits={self.task_cache_hits}, misses={self.task_cache_misses}"
                if self._cache_enabled()
                else ""
            )
        )
        return setting_results, setting_igd_curves, setting_hv_curves, setting_diagnostics

    def run_all(self):
        """运行所有算法、问题和重复次数的组合实验。"""

        (
            self.setting_results,
            self.setting_igd_curves,
            self.setting_hv_curves,
            self.setting_algorithm_diagnostics,
        ) = self._run_setting_sweep(
            self.cfg.ALGORITHMS,
            progress_prefix="RUN",
            collect_curves=True,
            collect_diagnostics=True,
        )
        canonical_key = self._canonical_setting_key()
        nt, tau_t = self._parse_setting_key(canonical_key)
        self.problems = DynamicTestProblems(nt=nt, tau_t=tau_t)
        self.results = self.setting_results.get(canonical_key, {})
        self.igd_curves = self.setting_igd_curves.get(canonical_key, {})
        self.hv_curves = self.setting_hv_curves.get(canonical_key, {})
        self.algorithm_diagnostics = self.setting_algorithm_diagnostics.get(canonical_key, {})
        return self.results

    def run_ablation_all(self):
        """运行默认消融对比。"""

        self.ablation_results = {}
        self.ablation_setting_results = {}
        if not getattr(self.cfg, "ABLATION_VARIANTS", None):
            return self.ablation_results

        print("\n  运行消融/对照变体...", flush=True)
        ablation_algorithms = {
            variant_name: {
                "algorithm": KEMM_DMOEA_Improved,
                "benchmark_aware_prior": bool(getattr(self.cfg, "ABLATION_BENCHMARK_PRIOR", False)),
                **variant_spec,
            }
            for variant_name, variant_spec in self.cfg.ABLATION_VARIANTS.items()
        }
        self.ablation_setting_results, _, _, _ = self._run_setting_sweep(
            ablation_algorithms,
            progress_prefix="ABL",
            collect_curves=False,
            collect_diagnostics=False,
        )
        self.ablation_results = self.ablation_setting_results.get(self._canonical_setting_key(), {})
        return self.ablation_results

    def _run_single(self, algo_spec, obj_func, pof_func, problem_suite=None, precomputed_aux: dict[str, object] | None = None):
        """运行一次单算法-单问题实验。

        这一步内部还包含多个环境变化阶段。
        在每次变化时：

        - 获得当前时间变量 `t`
        - 首次直接评价初始种群
        - 后续调用 `respond_to_change()`
        - 再执行若干代 `evolve_one_gen()`
        - 阶段结束后记录 `IGD / SP / MS`
        """

        lb = np.zeros(self.cfg.N_VAR)
        ub = np.ones(self.cfg.N_VAR)
        lb[1:] = -1.0
        ub[1:] = 1.0

        algo = self._instantiate_algorithm(algo_spec, lb, ub)
        algo.initialize()
        t0 = time.time()

        igd_list: List[float] = []
        hv_list: List[float] = []
        sp_list: List[float] = []
        ms_list: List[float] = []
        generation = 0
        problem_suite = problem_suite or self.problems
        precomputed_aux = dict(precomputed_aux or {})
        precomputed_times = [float(value) for value in precomputed_aux.get("times", [])]
        precomputed_true_pofs = [np.asarray(points, dtype=float) for points in precomputed_aux.get("true_pof_series", [])]

        for change_index in range(self.cfg.N_CHANGES):
            if change_index < len(precomputed_times):
                t = precomputed_times[change_index]
            else:
                t = problem_suite.get_time(generation)

            if change_index == 0:
                algo.fitness = algo.evaluate(algo.population, obj_func, t)
            else:
                algo.respond_to_change(obj_func, t)

            for _ in range(self.cfg.GENS_PER_CHANGE):
                algo.evolve_one_gen(obj_func, t)

            generation += self.cfg.GENS_PER_CHANGE
            obtained = algo.get_pareto_front()

            if change_index < len(precomputed_true_pofs):
                true_pof = precomputed_true_pofs[change_index]
            else:
                try:
                    true_pof = pof_func(t=t)
                except TypeError:
                    true_pof = pof_func()
            ref_point = np.max(np.vstack([true_pof, obtained]), axis=0) + 0.1

            igd_list.append(self.metrics.igd(obtained, true_pof))
            hv_list.append(self.metrics.hypervolume(obtained, ref_point))
            sp_list.append(self.metrics.spacing(obtained))
            ms_list.append(self.metrics.maximum_spread(obtained, true_pof))

        return {
            "migd": self.metrics.migd(igd_list),
            "sp": float(np.mean(sp_list)),
            "ms": float(np.mean(ms_list)),
            "time": time.time() - t0,
            "igd_curve": igd_list,
            "hv_curve": hv_list,
            "change_diagnostics": list(getattr(algo, "change_diagnostics_history", [])),
            "algo_instance": algo,
        }


def wilcoxon_test(ours: list, others: list, alpha: float = 0.05) -> str:
    """执行简化版 Wilcoxon/ranksums 显著性标记。

    返回：

    - `+`：ours 显著优于 others
    - `-`：ours 显著劣于 others
    - `≈`：无显著差异或样本不足
    """

    if len(ours) < 3:
        return "≈"
    try:
        from scipy.stats import ranksums

        _, p_value = ranksums(ours, others)
        if p_value < alpha:
            return "+" if np.mean(ours) < np.mean(others) else "-"
        return "≈"
    except Exception:
        return "≈"


class ResultPresenter:
    """benchmark 结果展示器。"""

    def __init__(
        self,
        results,
        config,
        igd_curves=None,
        hv_curves=None,
        algorithm_diagnostics=None,
        ablation_results=None,
        setting_results=None,
        ablation_setting_results=None,
    ):
        self.results = results
        self.cfg = config
        self.igd_curves = igd_curves or {}
        self.hv_curves = hv_curves or {}
        self.algorithm_diagnostics = algorithm_diagnostics or {}
        self.ablation_results = ablation_results or {}
        self.setting_results = setting_results or {}
        self.ablation_setting_results = ablation_setting_results or {}
        self.our_algo = "KEMM"

    def print_tables(self):
        """逐指标打印 benchmark 结果表。"""

        our = self.our_algo
        metrics_info = {"MIGD": "smaller", "SP": "smaller", "MS": "larger"}
        for metric, direction in metrics_info.items():
            arrow = "↓" if direction == "smaller" else "↑"
            print(f"\n{'=' * 110}")
            print(f"  TABLE: {metric} {arrow}  (Mean ± Std) [{self.cfg.N_RUNS} runs]")
            print(f"{'=' * 110}")
            algos = list(self.results.keys())
            header = f"{'Prob':>6s}"
            for algo in algos:
                header += f" | {algo:>16s}"
            print(header)
            print("-" * len(header))

            wins = defaultdict(int)
            for problem in self.cfg.PROBLEMS:
                row = f"{problem:>6s}"
                means = {algo: np.mean(self.results[algo][problem][metric]) for algo in algos}
                best = (min if direction == "smaller" else max)(means, key=means.get)
                wins[best] += 1
                for algo in algos:
                    values = self.results[algo][problem][metric]
                    mean_value = np.mean(values)
                    std_value = np.std(values)
                    if algo != our and our in self.results:
                        sig = wilcoxon_test(self.results[our][problem][metric], values)
                        if direction == "larger":
                            sig = {"+": "-", "-": "+", "≈": "≈"}[sig]
                    else:
                        sig = " "
                    mark = "**" if algo == best else "  "
                    row += f" | {mark}{mean_value:.4f}±{std_value:.4f}{sig}"
                print(row)

            print("-" * len(header))
            row = f"{'Wins':>6s}"
            for algo in algos:
                row += f" | {wins[algo]:>16d}"
            print(row)

    def print_ranking(self):
        """打印综合平均排名。"""

        print(f"\n{'=' * 60}")
        print("  综合排名 (MIGD+SP+MS 平均排名)")
        print(f"{'=' * 60}")
        algos = list(self.results.keys())
        all_ranks = {algo: [] for algo in algos}
        for metric in ["MIGD", "SP", "MS"]:
            direction = "smaller" if metric != "MS" else "larger"
            for problem in self.cfg.PROBLEMS:
                means = np.array([np.mean(self.results[algo][problem][metric]) for algo in algos])
                if direction == "larger":
                    means = -means
                ranks = np.argsort(np.argsort(means)) + 1
                for index, algo in enumerate(algos):
                    all_ranks[algo].append(ranks[index])

        avg_ranks = {algo: np.mean(rank_values) for algo, rank_values in all_ranks.items()}
        for rank, algo in enumerate(sorted(avg_ranks, key=avg_ranks.get), 1):
            marker = " ★ (KEMM 改进版)" if algo == self.our_algo else ""
            print(f"  #{rank}: {algo:>6s}  AvgRank = {avg_ranks[algo]:.2f}{marker}")

    def plot_mab_history(self, prefix="out"):
        """预留的 MAB 历史图接口。"""

        if not HAS_MPL:
            return
        print("  [INFO] 当前未从算法实例自动提取 MAB 历史数据。")

    def plot_all(self, prefix="out", plot_config=None):
        """导出 benchmark 侧最常用的一组图表。"""

        if not HAS_MPL:
            print("  [WARN] matplotlib 未安装")
            return

        from apps.reporting import BenchmarkFigurePayload, BenchmarkPlotConfig, generate_all_figures

        print("\n  生成可视化图表...")
        # 图表层只接收结构化 payload，而不是继续直接探测算法实例内部字段。
        # 这样后续即便你重构 KEMM 内部实现，只要 payload 结构保持稳定，报告层就不用跟着改。
        payload = BenchmarkFigurePayload(
            results=self.results,
            problems=self.cfg.PROBLEMS,
            igd_curves=self.igd_curves,
            hv_curves=self.hv_curves,
            diagnostics=self.algorithm_diagnostics,
            setting_results=self.setting_results,
            ablation_results=self.ablation_results,
            ablation_setting_results=self.ablation_setting_results,
            plot_config=plot_config if plot_config is not None else BenchmarkPlotConfig(),
        )
        generate_all_figures(payload=payload, output_prefix=prefix)
        print(f"  图表已保存 (前缀: {prefix}_*)")

    def _plot_metric_bars(self, prefix):
        algos = list(self.results.keys())
        n_algos = len(algos)
        base_colors = plt.cm.Set2(np.linspace(0, 1, n_algos))
        for metric, direction in [("MIGD", "smaller"), ("SP", "smaller"), ("MS", "larger")]:
            n_problems = len(self.cfg.PROBLEMS)
            n_cols = min(3, n_problems)
            n_rows = (n_problems + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
            axes = np.atleast_1d(axes).ravel()
            for idx, problem in enumerate(self.cfg.PROBLEMS):
                ax = axes[idx]
                means = [np.mean(self.results[algo][problem][metric]) for algo in algos]
                stds = [np.std(self.results[algo][problem][metric]) for algo in algos]
                best_index = int(np.argmin(means) if direction == "smaller" else np.argmax(means))
                colors = []
                for algo_index, algo in enumerate(algos):
                    if algo_index == best_index:
                        colors.append("gold")
                    elif algo == self.our_algo:
                        colors.append("#e74c3c")
                    else:
                        colors.append(base_colors[algo_index])
                ax.bar(range(n_algos), means, yerr=stds, capsize=3, color=colors, alpha=0.85, edgecolor="k", linewidth=0.5)
                ax.set_title(problem, fontweight="bold")
                ax.set_xticks(range(n_algos))
                ax.set_xticklabels(algos, rotation=45, fontsize=8)
                ax.grid(True, alpha=0.3, axis="y")
            for idx in range(n_problems, len(axes)):
                axes[idx].set_visible(False)
            fig.suptitle(f"{metric} Comparison (gold=best, red=KEMM-Improved)", fontsize=13, fontweight="bold")
            plt.tight_layout()
            plt.savefig(f"{prefix}_{metric.lower()}_bar.png", dpi=150, bbox_inches="tight")
            plt.close()

    def _plot_igd_over_time(self, prefix):
        if not self.igd_curves:
            return
        algos = list(self.results.keys())
        colors = plt.cm.Set1(np.linspace(0, 1, len(algos)))
        markers = ["o", "s", "^", "D", "v", "P", "*"]
        n_problems = len(self.cfg.PROBLEMS)
        n_cols = min(3, n_problems)
        n_rows = (n_problems + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4.5 * n_rows))
        axes = np.atleast_1d(axes).ravel()
        for idx, problem in enumerate(self.cfg.PROBLEMS):
            ax = axes[idx]
            for algo_index, algo in enumerate(algos):
                curves = self.igd_curves.get(algo, {}).get(problem, [])
                if not curves:
                    continue
                max_len = max(len(curve) for curve in curves)
                padded = [curve + [curve[-1]] * (max_len - len(curve)) for curve in curves]
                mean_curve = np.mean(padded, axis=0)
                std_curve = np.std(padded, axis=0)
                x = np.arange(1, len(mean_curve) + 1)
                linewidth = 2.5 if algo == self.our_algo else 1.2
                linestyle = "-" if algo == self.our_algo else "--"
                ax.plot(
                    x,
                    mean_curve,
                    marker=markers[algo_index % len(markers)],
                    linewidth=linewidth,
                    linestyle=linestyle,
                    markersize=5,
                    label=algo,
                    color=colors[algo_index],
                )
                ax.fill_between(x, mean_curve - std_curve, mean_curve + std_curve, color=colors[algo_index], alpha=0.08)
            ax.set_xlabel("Change Index")
            ax.set_ylabel("IGD")
            ax.set_title(problem, fontweight="bold")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7)
        for idx in range(n_problems, len(axes)):
            axes[idx].set_visible(False)
        fig.suptitle("IGD Over Time", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(f"{prefix}_igd_time.png", dpi=150, bbox_inches="tight")
        plt.close()

    def _plot_heatmap(self, prefix):
        algos = list(self.results.keys())
        matrix = np.array([[np.mean(self.results[algo][problem]["MIGD"]) for problem in self.cfg.PROBLEMS] for algo in algos])
        fig, ax = plt.subplots(figsize=(max(10, len(self.cfg.PROBLEMS) * 2), max(5, len(algos) * 0.8)))
        cmin = matrix.min(0, keepdims=True)
        cmax = matrix.max(0, keepdims=True)
        normalized = (matrix - cmin) / (cmax - cmin + 1e-12)
        im = ax.imshow(normalized, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=1)
        ax.set_xticks(range(len(self.cfg.PROBLEMS)))
        ax.set_xticklabels(self.cfg.PROBLEMS, fontsize=11)
        ax.set_yticks(range(len(algos)))
        ax.set_yticklabels(algos, fontsize=11)
        for i in range(len(algos)):
            for j in range(len(self.cfg.PROBLEMS)):
                text_color = "white" if normalized[i, j] > 0.6 else "black"
                ax.text(j, i, f"{matrix[i, j]:.4f}", ha="center", va="center", fontsize=9, color=text_color, fontweight="bold")
        plt.colorbar(im, ax=ax, label="Normalized MIGD (0=best)")
        ax.set_title("MIGD Heatmap (KEMM highlighted in tables)", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(f"{prefix}_heatmap.png", dpi=150, bbox_inches="tight")
        plt.close()

    def _plot_cd_diagram(self, prefix):
        algos = list(self.results.keys())
        all_ranks = {algo: [] for algo in algos}
        for metric in ["MIGD", "SP", "MS"]:
            direction = "smaller" if metric != "MS" else "larger"
            for problem in self.cfg.PROBLEMS:
                means = np.array([np.mean(self.results[algo][problem][metric]) for algo in algos])
                if direction == "larger":
                    means = -means
                ranks = np.argsort(np.argsort(means)).astype(float) + 1
                for index, algo in enumerate(algos):
                    all_ranks[algo].append(ranks[index])
        avg_ranks = {algo: np.mean(rank_values) for algo, rank_values in all_ranks.items()}
        sorted_algos = sorted(avg_ranks, key=avg_ranks.get)
        fig, ax = plt.subplots(figsize=(12, 4))
        colors = ["#e74c3c" if algo == self.our_algo else "#3498db" for algo in sorted_algos]
        ax.barh(range(len(sorted_algos)), [avg_ranks[algo] for algo in sorted_algos], color=colors, alpha=0.8, edgecolor="k", height=0.6)
        for index, algo in enumerate(sorted_algos):
            marker = " ★" if algo == self.our_algo else ""
            ax.text(avg_ranks[algo] + 0.05, index, f"{avg_ranks[algo]:.2f}{marker}", va="center", fontsize=11, fontweight="bold")
        ax.set_yticks(range(len(sorted_algos)))
        ax.set_yticklabels(sorted_algos, fontsize=12)
        ax.set_xlabel("Average Rank (lower=better)")
        ax.set_title("Average-Rank Comparison", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="x")
        plt.tight_layout()
        plt.savefig(f"{prefix}_cd_rank.png", dpi=150, bbox_inches="tight")
        plt.close()


def run_benchmark(
    quick: bool = False,
    with_jy: bool = False,
    output_dir: str | None = None,
    plot_config=None,
    workers: int | None = None,
    render_figures: bool = True,
    run_ablation: bool = True,
    algorithms: list[str] | None = None,
    problems: list[str] | None = None,
    force_rerun: bool = False,
):
    """benchmark 主线常用入口。"""

    print("=" * 70)
    print("  KEMM-DMOEA (改进版) 基准实验")
    print("  改进: 正确SGF + VAE记忆 + MAB算子选择 + GP漂移预测")
    print("=" * 70)

    cfg = ExperimentConfig()
    cfg.MAX_WORKERS = max(1, int(workers or 1))
    cfg.CACHE_ENABLED = True
    cfg.FORCE_RERUN = bool(force_rerun)
    cfg.CACHE_DIR = str(_BENCHMARK_TASK_CACHE_DIR)
    if quick:
        cfg.N_RUNS = 2
        cfg.N_CHANGES = 5
        cfg.GENS_PER_CHANGE = 10
        cfg.PROBLEMS = ["FDA1", "FDA3", "dMOP2"]
        print("  [MODE] 快速验证 (2次运行, 5次变化)")
    elif with_jy:
        cfg.PROBLEMS = cfg.PROBLEMS_STANDARD + cfg.PROBLEMS_JY
        print(f"  [MODE] 含 JY 系列 ({len(cfg.PROBLEMS)} 个测试函数)")

    if algorithms:
        unknown_algorithms = [name for name in algorithms if name not in cfg.ALGORITHMS]
        if unknown_algorithms:
            raise ValueError(f"Unknown benchmark algorithms: {unknown_algorithms}. Available: {list(cfg.ALGORITHMS.keys())}")
        cfg.ALGORITHMS = {name: cfg.ALGORITHMS[name] for name in algorithms}

    if problems:
        available_problems = list(dict.fromkeys(cfg.PROBLEMS_STANDARD + cfg.PROBLEMS_JY + list(cfg.PROBLEMS)))
        unknown_problems = [name for name in problems if name not in available_problems]
        if unknown_problems:
            raise ValueError(f"Unknown benchmark problems: {unknown_problems}. Available: {available_problems}")
        cfg.PROBLEMS = list(problems)

    print(f"\n  种群={cfg.POP_SIZE} 变量={cfg.N_VAR} 问题={cfg.PROBLEMS}")
    print(f"  运行次数={cfg.N_RUNS} (建议 ≥20 用于 Wilcoxon 检验)")
    print(f"  并行进程={cfg.MAX_WORKERS}")
    print(f"  导出图表={'是' if render_figures else '否'}  消融实验={'是' if run_ablation else '否'}")
    print(
        f"  任务缓存={'开启' if cfg.CACHE_ENABLED else '关闭'}"
        + (f"  强制重跑={'是' if cfg.FORCE_RERUN else '否'}" if cfg.CACHE_ENABLED else "")
    )
    print(f"  算法: {list(cfg.ALGORITHMS.keys())}\n")

    report_root = build_report_paths(Path(output_dir) if output_dir else None, prefix="benchmark")
    (report_root / "figures").mkdir(parents=True, exist_ok=True)
    figures_prefix = str(report_root / "figures" / "benchmark")

    runner = ExperimentRunner(cfg)
    results = runner.run_all()
    ablation_results = runner.run_ablation_all() if run_ablation else {}

    presenter = ResultPresenter(
        results,
        cfg,
        igd_curves=runner.igd_curves,
        hv_curves=runner.hv_curves,
        algorithm_diagnostics=runner.algorithm_diagnostics,
        ablation_results=ablation_results,
        setting_results=runner.setting_results,
        ablation_setting_results=runner.ablation_setting_results,
    )
    print()
    presenter.print_tables()
    presenter.print_ranking()
    if render_figures:
        presenter.plot_all(prefix=figures_prefix, plot_config=plot_config)
    else:
        print("  [SKIP] 已跳过 benchmark 图表导出，仅保留原始结果与 Markdown 摘要。")
    export_benchmark_report(
        results,
        cfg,
        output_root=report_root,
        ablation_results=ablation_results,
        setting_results=runner.setting_results,
        ablation_setting_results=runner.ablation_setting_results,
    )
    print(f"\n  报告输出目录: {report_root}")
    return results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run benchmark experiments.")
    parser.add_argument("--quick", action="store_true", help="Run a lightweight smoke configuration.")
    parser.add_argument("--full", action="store_true", help="Run the default full benchmark suite.")
    parser.add_argument("--with-jy", action="store_true", help="Append JY problems to the benchmark suite.")
    parser.add_argument("--workers", type=int, default=None, help="Process workers for outer-run parallelism; default is auto for full mode and 1 for quick mode.")
    parser.add_argument("--skip-ablation", action="store_true", help="Skip ablation/control variants to reduce runtime.")
    parser.add_argument("--summary-only", action="store_true", help="Skip figure rendering and export only raw tables plus Markdown summary.")
    parser.add_argument("--force-rerun", action="store_true", help="Ignore benchmark task cache and recompute all tasks for this run.")
    parser.add_argument("--algorithms", nargs="*", default=None, help="Optional benchmark algorithms to run, e.g. KEMM RI Tr.")
    parser.add_argument("--problems", nargs="*", default=None, help="Optional benchmark problems to run, e.g. FDA1 FDA3 dMOP2.")
    parser.add_argument("--output-dir", default=None, help="Optional output directory.")
    parser.add_argument("--plot-preset", default="paper", help="Plot preset: default/paper/ieee/nature/thesis.")
    parser.add_argument("--science-style", default="", help="Comma-separated SciencePlots style tuple.")
    parser.add_argument("--appendix-plots", action="store_true", help="Export appendix/debug benchmark plots.")
    parser.add_argument("--interactive-figures", action="store_true", help="Also export interactive matplotlib figure bundles (.fig.pickle).")
    return parser.parse_args()


def main():
    """命令行入口。"""

    args = _parse_args()
    style_overrides = {}
    if args.science_style:
        style_overrides["use_scienceplots"] = True
        style_overrides["science_styles"] = tuple(part.strip() for part in args.science_style.split(",") if part.strip())
    plot_config = build_benchmark_plot_config(
        preset=args.plot_preset,
        style_overrides=style_overrides,
        appendix_plots=args.appendix_plots,
        interactive_figures=args.interactive_figures,
    )
    workers = args.workers
    if workers is None:
        workers = 1 if args.quick and not args.full and not args.with_jy else _recommended_worker_count()
    render_figures = not args.summary_only
    run_ablation = not args.skip_ablation

    if args.full:
        run_benchmark(
            quick=False,
            output_dir=args.output_dir,
            plot_config=plot_config,
            workers=workers,
            render_figures=render_figures,
            run_ablation=run_ablation,
            algorithms=args.algorithms,
            problems=args.problems,
            force_rerun=args.force_rerun,
        )
    elif args.with_jy:
        run_benchmark(
            quick=False,
            with_jy=True,
            output_dir=args.output_dir,
            plot_config=plot_config,
            workers=workers,
            render_figures=render_figures,
            run_ablation=run_ablation,
            algorithms=args.algorithms,
            problems=args.problems,
            force_rerun=args.force_rerun,
        )
    elif args.quick:
        run_benchmark(
            quick=True,
            output_dir=args.output_dir,
            plot_config=plot_config,
            workers=workers,
            render_figures=render_figures,
            run_ablation=run_ablation,
            algorithms=args.algorithms,
            problems=args.problems,
            force_rerun=args.force_rerun,
        )
    else:
        run_benchmark(
            quick=True,
            output_dir=args.output_dir,
            plot_config=plot_config,
            workers=workers,
            render_figures=render_figures,
            run_ablation=run_ablation,
            algorithms=args.algorithms,
            problems=args.problems,
            force_rerun=args.force_rerun,
        )
        print(
            "\n  使用选项: --quick | --full | --with-jy | --workers <n> | --skip-ablation | --force-rerun | "
            "--summary-only | --algorithms <names...> | --problems <names...> | "
            "--output-dir <path> | --plot-preset <preset>"
        )


if __name__ == "__main__":
    main()
