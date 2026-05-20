import numpy as np
import itertools
from core import config as base_cfg
from core import data as data_module
from algorithms.ga import GeneticAlgorithm
from algorithms.sa import SimulatedAnnealing


def _make_fields_data():
    return {
        "demand_min": data_module.get_demand_min(),
        "demand_max": data_module.get_demand_max(),
        "prices":     data_module.get_prices(),
        "W_total":    base_cfg.W_TOTAL,
    }


class _DynamicConfig:
    def __init__(self, **kwargs):
        import core.config as c
        for k, v in vars(c).items():
            if not k.startswith("__"):
                setattr(self, k, v)
        for k, v in kwargs.items():
            setattr(self, k, v)


def grid_search_ga(param_grid: dict, fields_data: dict = None, n_runs: int = 3) -> dict:
    if fields_data is None:
        fields_data = _make_fields_data()
    keys   = list(param_grid.keys())
    values = list(param_grid.values())
    best_params = None
    best_mean   = float("inf")
    results_grid = {}
    for combo in itertools.product(*values):
        params = dict(zip(keys, combo))
        cfg = _DynamicConfig(**params)
        fitnesses = []
        for _ in range(n_runs):
            ga = GeneticAlgorithm(fields_data, cfg)
            ga.solve()
            fitnesses.append(ga.best_fitness)
        mean = float(np.mean(fitnesses))
        results_grid[str(params)] = mean
        if mean < best_mean:
            best_mean   = mean
            best_params = params
    return {"best_params": best_params, "best_mean": best_mean, "grid": results_grid}


def grid_search_sa(param_grid: dict, fields_data: dict = None, n_runs: int = 3) -> dict:
    if fields_data is None:
        fields_data = _make_fields_data()
    keys   = list(param_grid.keys())
    values = list(param_grid.values())
    best_params = None
    best_mean   = float("inf")
    results_grid = {}
    for combo in itertools.product(*values):
        params = dict(zip(keys, combo))
        cfg = _DynamicConfig(**params)
        fitnesses = []
        for _ in range(n_runs):
            sa = SimulatedAnnealing(fields_data, cfg)
            sa.solve()
            fitnesses.append(sa.best_fitness)
        mean = float(np.mean(fitnesses))
        results_grid[str(params)] = mean
        if mean < best_mean:
            best_mean   = mean
            best_params = params
    return {"best_params": best_params, "best_mean": best_mean, "grid": results_grid}


def get_heatmap_data(results_grid: dict, param1_values, param2_values) -> np.ndarray:
    data = np.zeros((len(param1_values), len(param2_values)))
    for i, v1 in enumerate(param1_values):
        for j, v2 in enumerate(param2_values):
            key_match = [k for k in results_grid if str(v1) in k and str(v2) in k]
            if key_match:
                data[i, j] = results_grid[key_match[0]]
    return data
