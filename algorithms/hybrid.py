import numpy as np
import math
from core.constraints import repair_solution, clip_to_bounds
from algorithms.base import BaseAlgorithm


class HybridGASA(BaseAlgorithm):
    """GA chạy trước để tìm vùng tốt, sau đó SA tinh chỉnh nghiệm tốt nhất."""

    def __init__(self, fields_data: dict, config):
        super().__init__(fields_data, config)
        self.pop_size  = config.GA_POP_SIZE
        self.ga_gen    = config.HYBRID_GA_GEN
        self.cx_rate   = config.GA_CROSSOVER_RATE
        self.mut_rate  = config.GA_MUTATION_RATE
        self.mut_sigma = config.GA_MUTATION_SIGMA
        self.tourn_k   = config.GA_TOURNAMENT_K
        self.elite_n   = max(1, int(config.GA_ELITE_RATIO * config.GA_POP_SIZE))
        self.blx_alpha = config.GA_BLX_ALPHA
        self.T_max     = config.HYBRID_SA_T_MAX
        self.T_min     = config.SA_T_MIN
        self.sa_alpha  = config.SA_ALPHA
        self.iter_per_T = config.SA_ITER_PER_T
        self.sa_sigma  = config.SA_NEIGHBOR_SIGMA

    def _tournament(self, pop, fits):
        idx  = np.random.choice(len(pop), self.tourn_k, replace=False)
        best = idx[np.argmin([fits[i] for i in idx])]
        return pop[best].copy()

    def _blx(self, p1, p2):
        d  = np.abs(p1 - p2)
        lo = np.minimum(p1, p2) - self.blx_alpha * d
        hi = np.maximum(p1, p2) + self.blx_alpha * d
        c  = np.random.uniform(lo, hi)
        return repair_solution(
            clip_to_bounds(c, self.demand_min, self.demand_max),
            self.demand_min, self.demand_max, self.W_total
        )

    def _mutate(self, x):
        c = x.copy()
        for i in range(self.n_fields):
            if np.random.rand() < self.mut_rate:
                c[i] += np.random.normal(0, self.mut_sigma)
        return repair_solution(
            clip_to_bounds(c, self.demand_min, self.demand_max),
            self.demand_min, self.demand_max, self.W_total
        )

    def _run_ga(self):
        pop  = [self.random_solution() for _ in range(self.pop_size)]
        fits = [self.fitness(x) for x in pop]

        for _ in range(self.ga_gen):
            order  = np.argsort(fits)
            elites = [pop[i].copy() for i in order[:self.elite_n]]
            new_pop = elites[:]
            while len(new_pop) < self.pop_size:
                p1    = self._tournament(pop, fits)
                p2    = self._tournament(pop, fits)
                child = self._blx(p1, p2) if np.random.rand() < self.cx_rate else p1.copy()
                child = self._mutate(child)
                new_pop.append(child)
            pop  = new_pop
            fits = [self.fitness(x) for x in pop]
            self.history.append(min(fits))

        best_idx = int(np.argmin(fits))
        return pop[best_idx].copy(), fits[best_idx]

    def _run_sa(self, start):
        current   = start.copy()
        f_current = self.fitness(current)
        T = self.T_max
        while T > self.T_min:
            for _ in range(self.iter_per_T):
                idx = np.random.randint(self.n_fields)
                nb  = current.copy()
                nb[idx] += np.random.normal(0, self.sa_sigma)
                nb = repair_solution(
                    clip_to_bounds(nb, self.demand_min, self.demand_max),
                    self.demand_min, self.demand_max, self.W_total
                )
                f_nb  = self.fitness(nb)
                delta = f_nb - f_current
                if delta < 0 or np.random.rand() < math.exp(-delta / T):
                    current   = nb
                    f_current = f_nb
                if f_current < self.best_fitness:
                    self.best_fitness  = f_current
                    self.best_solution = current.copy()
            T *= self.sa_alpha
            self.history.append(self.best_fitness)
        return current

    def solve(self) -> np.ndarray:
        ga_best, ga_fit    = self._run_ga()
        self.best_solution = ga_best.copy()
        self.best_fitness  = ga_fit
        self._run_sa(ga_best)
        return self.best_solution
