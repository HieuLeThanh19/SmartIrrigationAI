# algorithms/ga.py — Genetic Algorithm

import numpy as np
from core.constraints import repair_solution, clip_to_bounds
from algorithms.base import BaseAlgorithm


class GeneticAlgorithm(BaseAlgorithm):
    def __init__(self, fields_data: dict, config):
        super().__init__(fields_data, config)
        self.pop_size   = config.GA_POP_SIZE
        self.max_gen    = config.GA_MAX_GEN
        self.cx_rate    = config.GA_CROSSOVER_RATE
        self.mut_rate   = config.GA_MUTATION_RATE
        self.mut_sigma  = config.GA_MUTATION_SIGMA
        self.tourn_k    = config.GA_TOURNAMENT_K
        self.elite_n    = max(1, int(config.GA_ELITE_RATIO * config.GA_POP_SIZE))
        self.blx_alpha  = config.GA_BLX_ALPHA

    def _init_pop(self):
        return [self.random_solution() for _ in range(self.pop_size)]

    def _tournament(self, pop, fits):
        idx = np.random.choice(len(pop), self.tourn_k, replace=False)
        best = idx[np.argmin([fits[i] for i in idx])]
        return pop[best].copy()

    def _blx_crossover(self, p1, p2):
        d = np.abs(p1 - p2)
        lo = np.minimum(p1, p2) - self.blx_alpha * d
        hi = np.maximum(p1, p2) + self.blx_alpha * d
        child = np.random.uniform(lo, hi)
        return repair_solution(
            clip_to_bounds(child, self.demand_min, self.demand_max),
            self.demand_min, self.demand_max, self.W_total
        )

    def _mutate(self, x):
        child = x.copy()
        for i in range(self.n_fields):
            if np.random.rand() < self.mut_rate:
                child[i] += np.random.normal(0, self.mut_sigma)
        return repair_solution(
            clip_to_bounds(child, self.demand_min, self.demand_max),
            self.demand_min, self.demand_max, self.W_total
        )

    def solve(self) -> np.ndarray:
        pop  = self._init_pop()
        fits = [self.fitness(x) for x in pop]

        for x, f in zip(pop, fits):
            if f < self.best_fitness:
                self.best_fitness  = f
                self.best_solution = x.copy()

        for _ in range(self.max_gen):
            order = np.argsort(fits)
            elites = [pop[i].copy() for i in order[:self.elite_n]]

            new_pop = elites[:]
            while len(new_pop) < self.pop_size:
                p1 = self._tournament(pop, fits)
                p2 = self._tournament(pop, fits)
                child = self._blx_crossover(p1, p2) if np.random.rand() < self.cx_rate else p1.copy()
                child = self._mutate(child)
                new_pop.append(child)

            pop  = new_pop
            fits = [self.fitness(x) for x in pop]

            gen_best = min(fits)
            if gen_best < self.best_fitness:
                self.best_fitness  = gen_best
                self.best_solution = pop[np.argmin(fits)].copy()
            self.history.append(self.best_fitness)

        return self.best_solution
