import numpy as np


def clip_to_bounds(x: np.ndarray, demand_min: np.ndarray,
                   demand_max: np.ndarray) -> np.ndarray:
    return np.clip(x, demand_min, demand_max)


def is_feasible(x: np.ndarray, demand_min: np.ndarray,
                demand_max: np.ndarray) -> bool:
    return bool(np.all(x >= demand_min) and np.all(x <= demand_max))


def water_budget_violation(x: np.ndarray, W_total: float) -> float:
    return float(max(0.0, np.sum(x) - W_total))


def repair_solution(x: np.ndarray, demand_min: np.ndarray,
                    demand_max: np.ndarray, W_total: float) -> np.ndarray:
    """Clip về bounds, rồi giảm phần vượt ngân sách trên phần dư so với demand_min."""
    x = clip_to_bounds(x, demand_min, demand_max)
    total = np.sum(x)
    if total <= W_total:
        return x

    min_total = float(np.sum(demand_min))
    if min_total > W_total:
        # Bài toán vô nghiệm nếu tổng min vượt ngân sách. Trả về phương án tôn trọng ngân sách
        # để thuật toán không vỡ số; giao diện sẽ chặn input này trước khi chạy.
        return demand_min * (W_total / min_total)

    reducible = x - demand_min
    reducible_total = float(np.sum(reducible))
    if reducible_total <= 0:
        return demand_min.copy()

    reduction = min(float(total - W_total), reducible_total)
    x = x - reducible * (reduction / reducible_total)
    return x
