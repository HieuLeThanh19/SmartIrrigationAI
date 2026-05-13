# analysis/comparator.py — So sánh kết quả thuật toán
import numpy as np
import pandas as pd


def build_summary_table(results: dict) -> pd.DataFrame:
    rows = []
    for algo, runs in results.items():
        fits = [r["best_fitness"] for r in runs]
        rts = [r.get("runtime", 0.0) for r in runs]
        rows.append(
            {
                "Thuật toán": algo,
                "Điểm tốt nhất": round(min(fits) if fits else 0.0, 4),
                "Điểm trung bình": round(float(np.mean(fits)) if fits else 0.0, 4),
                "Độ lệch chuẩn": round(float(np.std(fits)) if fits else 0.0, 4),
                "Thời gian (s)": round(float(np.mean(rts)) if rts else 0.0, 3),
                "Số lần chạy": len(runs),
            }
        )
    table = pd.DataFrame(rows)
    if not table.empty:
        table = table.sort_values(by="Điểm trung bình", ascending=True).reset_index(drop=True)
    return table


def rank_algorithms(results: dict) -> list:
    mean_fits = {
        a: float(np.mean([r["best_fitness"] for r in runs])) if runs else float("inf")
        for a, runs in results.items()
    }
    return sorted(mean_fits, key=mean_fits.get)


def compute_improvement(results: dict, baseline: str = "GA") -> dict:
    if baseline not in results or not results[baseline]:
        return {}
    base_mean = float(np.mean([r["best_fitness"] for r in results[baseline]]))
    out = {}
    for algo, runs in results.items():
        mean = float(np.mean([r["best_fitness"] for r in runs])) if runs else base_mean
        out[algo] = round((base_mean - mean) / base_mean * 100, 2) if base_mean > 0 else 0.0
    return out


def get_best_allocation(results: dict) -> dict:
    best_run = None
    best_algo = None
    best_fitness = float("inf")
    for algo, runs in results.items():
        for r in runs:
            if r["best_fitness"] < best_fitness:
                best_fitness = r["best_fitness"]
                best_run = r
                best_algo = algo
    if best_run is None:
        return {}
    return {
        "algo_name": best_algo,
        "fitness": float(best_fitness),
        "solution": np.array(best_run["best_solution"], dtype=float),
    }
