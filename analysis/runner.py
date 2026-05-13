# analysis/runner.py — Chạy thực nghiệm tự động
import json
import time
from pathlib import Path
from core import config as cfg
from core.data import get_demand_min, get_demand_max, get_prices, get_field_names, get_crop_names
from algorithms.ga     import GeneticAlgorithm
from algorithms.sa     import SimulatedAnnealing
from algorithms.pso    import ParticleSwarmOptimization
from algorithms.hybrid import HybridGASA


ALGO_MAP = {
    "GA":     GeneticAlgorithm,
    "SA":     SimulatedAnnealing,
    "PSO":    ParticleSwarmOptimization,
    "Hybrid": HybridGASA,
}


class ExperimentRunner:
    def __init__(self, algorithms_list: list, fields_data: dict,
                 config=cfg, n_runs: int = 10):
        self.algorithms_list = algorithms_list
        self.fields_data     = fields_data
        self.config          = config
        self.n_runs          = n_runs
        self.results: dict   = {}

    def run_single(self, algo_name: str, run_id: int) -> dict:
        if algo_name not in ALGO_MAP:
            raise ValueError(f"Unknown algorithm: {algo_name}")
        cls   = ALGO_MAP[algo_name]
        algo  = cls(self.fields_data, self.config)
        t0    = time.perf_counter()
        sol   = algo.solve()
        rt    = time.perf_counter() - t0
        return {
            "algo_name": algo_name,
            "run_id":       run_id,
            "best_fitness": float(algo.get_best_fitness()),
            "best_solution": sol.tolist(),
            "history":      algo.get_history(),
            "runtime":      round(rt, 4),
        }

    def run_all(self, callback=None) -> dict:
        total = len(self.algorithms_list) * self.n_runs
        done = 0
        for algo_name in self.algorithms_list:
            self.results[algo_name] = []
            for run_id in range(self.n_runs):
                r = self.run_single(algo_name, run_id)
                self.results[algo_name].append(r)
                done += 1
                if callback:
                    callback(algo_name, run_id + 1, self.n_runs, done, total)
        return self.results

    def get_best_per_algo(self) -> dict:
        return {
            a: min(runs, key=lambda r: r["best_fitness"])["best_fitness"]
            for a, runs in self.results.items()
        }

    def save_results(self, filepath: str):
        output = Path(filepath)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)

    def load_results(self, filepath: str):
        with open(filepath, "r", encoding="utf-8") as f:
            self.results = json.load(f)
        return self.results


def build_fields_data(W_total=None) -> dict:
    """Tạo fields_data chuẩn từ core/data.py."""
    return {
        "demand_min":  get_demand_min(),
        "demand_max":  get_demand_max(),
        "prices":      get_prices(),
        "W_total":     W_total or cfg.W_TOTAL,
        "field_names": get_field_names(),
        "crop_names":  get_crop_names(),
    }
