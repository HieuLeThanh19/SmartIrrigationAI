# algorithms/base.py — Lớp cơ sở cho tất cả thuật toán

import numpy as np
from abc import ABC, abstractmethod
from core.constraints import repair_solution
from core.fitness import fitness as calculate_fitness


class BaseAlgorithm(ABC):
    def __init__(self, fields_data: dict, config):
        self.demand_min  = np.array(fields_data["demand_min"])
        self.demand_max  = np.array(fields_data["demand_max"])
        self.prices      = np.array(fields_data["prices"])
        self.W_total     = fields_data["W_total"]
        self.n_fields    = len(self.demand_min)
        self.config      = config

        self.best_solution = None
        self.best_fitness  = float("inf")
        self.history       = []  # fitness tốt nhất theo từng bước

    def fitness(self, x: np.ndarray) -> float:
        return calculate_fitness(x, self.demand_min, self.demand_max, self.prices, self.W_total)

    def random_solution(self) -> np.ndarray:
        x = np.random.uniform(self.demand_min, self.demand_max)
        return repair_solution(x, self.demand_min, self.demand_max, self.W_total)

    def update_best(self, x: np.ndarray):
        f = self.fitness(x)
        if f < self.best_fitness:
            self.best_fitness = f
            self.best_solution = x.copy()

    def get_best_fitness(self) -> float:
        return self.best_fitness

    def get_history(self) -> list:
        return self.history

    @abstractmethod
    def solve(self) -> np.ndarray:
        pass
