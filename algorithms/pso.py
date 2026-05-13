# algorithms/pso.py — Particle Swarm Optimization

import numpy as np
from core.constraints import repair_solution, clip_to_bounds
from algorithms.base import BaseAlgorithm


class ParticleSwarmOptimization(BaseAlgorithm):
    def __init__(self, fields_data: dict, config):
        super().__init__(fields_data, config)
        self.n_particles = config.PSO_N_PARTICLES
        self.max_iter    = config.PSO_MAX_ITER
        self.w           = config.PSO_W
        self.c1          = config.PSO_C1
        self.c2          = config.PSO_C2
        self.v_max       = config.PSO_V_MAX

    def solve(self) -> np.ndarray:
        n = self.n_particles
        d = self.n_fields

        pos = np.array([self.random_solution() for _ in range(n)])
        vel = np.random.uniform(-self.v_max, self.v_max, (n, d))

        pbest     = pos.copy()
        pbest_fit = np.array([self.fitness(p) for p in pbest])

        gbest_idx = np.argmin(pbest_fit)
        gbest     = pbest[gbest_idx].copy()
        gbest_fit = pbest_fit[gbest_idx]

        self.best_solution = gbest.copy()
        self.best_fitness  = gbest_fit

        for _ in range(self.max_iter):
            r1 = np.random.rand(n, d)
            r2 = np.random.rand(n, d)

            vel = (self.w * vel
                   + self.c1 * r1 * (pbest - pos)
                   + self.c2 * r2 * (gbest - pos))
            vel = np.clip(vel, -self.v_max, self.v_max)
            pos = pos + vel

            for i in range(n):
                pos[i] = repair_solution(
                    clip_to_bounds(pos[i], self.demand_min, self.demand_max),
                    self.demand_min, self.demand_max, self.W_total
                )
                f = self.fitness(pos[i])
                if f < pbest_fit[i]:
                    pbest[i]     = pos[i].copy()
                    pbest_fit[i] = f
                if f < gbest_fit:
                    gbest     = pos[i].copy()
                    gbest_fit = f

            self.best_solution = gbest.copy()
            self.best_fitness  = gbest_fit
            self.history.append(self.best_fitness)

        return self.best_solution
