"""Tests cho core/constraints.py"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from core.constraints import clip_to_bounds, repair_solution, water_budget_violation
from core.data import get_demand_min, get_demand_max
from core import config as cfg

DM = get_demand_min()
DX = get_demand_max()


def test_clip_to_bounds():
    x = DM - 10
    clipped = clip_to_bounds(x, DM, DX)
    assert np.all(clipped >= DM)
    x2 = DX + 10
    clipped2 = clip_to_bounds(x2, DM, DX)
    assert np.all(clipped2 <= DX)


def test_repair_solution_budget():
    x = DX.copy()  # sum > W_total
    repaired = repair_solution(x, DM, DX, cfg.W_TOTAL)
    assert np.sum(repaired) <= cfg.W_TOTAL + 1e-6, "Vẫn vi phạm ngân sách sau repair"
    assert np.all(repaired >= DM - 1e-6), "Vi phạm demand_min sau repair"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
