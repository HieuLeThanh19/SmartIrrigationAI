# core/fitness.py — Hàm tính fitness trung tâm của project
import numpy as np
from core import config as cfg


def penalty_water_shortage(x: np.ndarray, demand_min: np.ndarray) -> float:
    """Penalty thiếu nước (bình phương để phạt nặng hơn khi thiếu nhiều)."""
    diff = np.maximum(0.0, demand_min - x)
    return float(np.sum(diff ** 2))


def penalty_water_waste(x: np.ndarray, demand_max: np.ndarray) -> float:
    """Penalty lãng phí (tuyến tính)."""
    diff = np.maximum(0.0, x - demand_max)
    return float(np.sum(diff))


def penalty_water_over_budget(x: np.ndarray, W_total: float) -> float:
    """Penalty khi tổng lượng nước vượt ngân sách cho phép."""
    return float(max(0.0, np.sum(x) - W_total))


def pumping_cost(x: np.ndarray, prices: np.ndarray) -> float:
    """Chi phí bơm nước."""
    return float(np.dot(prices, x))


def fitness(x: np.ndarray, demand_min: np.ndarray,
            demand_max: np.ndarray, prices: np.ndarray,
            W_total: float = None) -> float:
    """
    Hàm fitness tổng hợp (minimize).
    fitness = ALPHA * thiếu_nước + BETA * (dư_từng_thửa + vượt_tổng_nước) + GAMMA * chi_phí_bơm
    """
    shortage = penalty_water_shortage(x, demand_min)
    waste    = penalty_water_waste(x, demand_max)
    over_budget = penalty_water_over_budget(x, W_total) if W_total is not None else 0.0
    cost     = pumping_cost(x, prices)
    return cfg.ALPHA_PENALTY * shortage + cfg.BETA_WASTE * (waste + over_budget) + cfg.GAMMA_COST * cost


def fitness_breakdown(x: np.ndarray, demand_min: np.ndarray,
                      demand_max: np.ndarray, prices: np.ndarray,
                      W_total: float = None) -> dict:
    """Trả về dict chi tiết các thành phần fitness để hiển thị trên giao diện."""
    shortage = penalty_water_shortage(x, demand_min)
    waste    = penalty_water_waste(x, demand_max)
    over_budget = penalty_water_over_budget(x, W_total) if W_total is not None else 0.0
    cost     = pumping_cost(x, prices)
    shortage_penalty = cfg.ALPHA_PENALTY * shortage
    waste_penalty = cfg.BETA_WASTE * (waste + over_budget)
    cost_score = cfg.GAMMA_COST * cost
    total = shortage_penalty + waste_penalty + cost_score
    return {
        "shortage":         float(shortage),
        "waste":            float(waste),
        "over_budget":      float(over_budget),
        "cost_raw":         float(cost),
        "shortage_penalty": float(shortage_penalty),
        "waste_penalty":    float(waste_penalty),
        "cost":             float(cost_score),
        "total":            float(total),
    }
