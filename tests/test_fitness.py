"""Tests cho core/fitness.py"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from core.fitness import fitness, penalty_water_shortage, penalty_water_waste, fitness_breakdown
from core.data import get_demand_min, get_demand_max, get_prices
from core import config as cfg

DM = get_demand_min()
DX = get_demand_max()
PR = get_prices()


def test_fitness_all_at_min():
    x = DM.copy()
    shortage = penalty_water_shortage(x, DM)
    assert shortage == pytest.approx(0.0), "Không được có penalty khi x=min"


def test_fitness_shortage():
    x = DM.copy()
    x[0] = DM[0] - 10  # thiếu 10 đơn vị
    shortage = penalty_water_shortage(x, DM)
    assert shortage == pytest.approx(100.0), "Penalty phải = 10² = 100"


def test_fitness_waste():
    x = DX.copy()
    x[0] = DX[0] + 5  # dư 5 đơn vị
    waste = penalty_water_waste(x, DX)
    assert waste == pytest.approx(5.0)


def test_fitness_breakdown_sum():
    x = DM + (DX - DM) * 0.5  # midpoint
    bd = fitness_breakdown(x, DM, DX, PR)
    expected = bd["shortage_penalty"] + bd["waste_penalty"] + bd["cost"]
    assert bd["total"] == pytest.approx(expected)


def test_fitness_matches_breakdown():
    x = DM + (DX - DM) * 0.5
    bd = fitness_breakdown(x, DM, DX, PR, cfg.W_TOTAL)
    assert fitness(x, DM, DX, PR, cfg.W_TOTAL) == pytest.approx(bd["total"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
