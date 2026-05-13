# algorithms/sa.py — Simulated Annealing

import numpy as np
import math
from core.constraints import repair_solution, clip_to_bounds
from algorithms.base import BaseAlgorithm


class SimulatedAnnealing(BaseAlgorithm):
    def __init__(self, fields_data: dict, config):
        super().__init__(fields_data, config)
        self.T_max    = config.SA_T_MAX
        self.T_min    = config.SA_T_MIN
        self.alpha    = config.SA_ALPHA
        self.iter_per_T = config.SA_ITER_PER_T
        self.sigma    = config.SA_NEIGHBOR_SIGMA

    def _neighbor(self, x):
        idx = np.random.randint(self.n_fields)
        nb  = x.copy()
        nb[idx] += np.random.normal(0, self.sigma)
        return repair_solution(
            clip_to_bounds(nb, self.demand_min, self.demand_max),
            self.demand_min, self.demand_max, self.W_total
        )

    def solve(self) -> np.ndarray:
        current   = self.random_solution()
        f_current = self.fitness(current)
        self.best_solution = current.copy()
        self.best_fitness  = f_current

        T = self.T_max
        while T > self.T_min:
            for _ in range(self.iter_per_T):
                nb   = self._neighbor(current)
                f_nb = self.fitness(nb)
                delta = f_nb - f_current
                if delta < 0 or np.random.rand() < math.exp(-delta / T):
                    current   = nb
                    f_current = f_nb
                if f_current < self.best_fitness:
                    self.best_fitness  = f_current
                    self.best_solution = current.copy()
            T *= self.alpha
            self.history.append(self.best_fitness)

        return self.best_solution
