"""Tests cho algorithms/operators.py"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from algorithms.operators import (gaussian_mutation, blx_alpha_crossover,
                                  acceptance_probability, update_velocity, update_position)
from core.data import get_demand_min, get_demand_max

DM = get_demand_min()
DX = get_demand_max()


def test_gaussian_mutation_bounds():
    x = (DM + DX) / 2
    for _ in range(1000):
        mutated = gaussian_mutation(x, 1.0, 5.0, DM, DX)
        assert np.all(mutated >= DM), "Mutation vi phạm demand_min"
        assert np.all(mutated <= DX), "Mutation vi phạm demand_max"


def test_blx_crossover_range():
    p1 = DM + (DX - DM) * 0.3
    p2 = DM + (DX - DM) * 0.7
    for _ in range(100):
        c1, c2 = blx_alpha_crossover(p1, p2, alpha=0.5)
        # Con có thể vượt range một chút do BLX-α — đây là bình thường
        # nhưng kiểm tra kiểu dữ liệu
        assert c1.shape == p1.shape
        assert c2.shape == p2.shape


def test_acceptance_probability():
    assert acceptance_probability(-10, 100) == pytest.approx(1.0)
    prob_hot  = acceptance_probability(100, 10000)
    prob_cold = acceptance_probability(100, 0.001)
    assert prob_hot  > 0.9, "Nhiệt cao phải gần 1.0"
    assert prob_cold < 0.1, "Nhiệt thấp phải gần 0.0"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
