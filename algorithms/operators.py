"""Reusable operators for GA, SA, and PSO experiments.

The main algorithm classes keep their own small helper methods, but these
functions are useful for tests, notebooks, and parameter experiments.
"""

import math

import numpy as np

from core.constraints import clip_to_bounds


def gaussian_mutation(
    x: np.ndarray,
    mutation_rate: float,
    sigma: float,
    demand_min: np.ndarray,
    demand_max: np.ndarray,
) -> np.ndarray:
    """Mutate each dimension with Gaussian noise, then keep values in bounds."""
    child = np.array(x, dtype=float, copy=True)
    mask = np.random.rand(child.size) < mutation_rate
    child[mask] += np.random.normal(0.0, sigma, size=int(np.sum(mask)))
    return clip_to_bounds(child, demand_min, demand_max)


def blx_alpha_crossover(
    parent1: np.ndarray,
    parent2: np.ndarray,
    alpha: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Create two BLX-alpha children from two parents."""
    p1 = np.array(parent1, dtype=float)
    p2 = np.array(parent2, dtype=float)
    delta = np.abs(p1 - p2)
    low = np.minimum(p1, p2) - alpha * delta
    high = np.maximum(p1, p2) + alpha * delta
    return np.random.uniform(low, high), np.random.uniform(low, high)


def acceptance_probability(delta: float, temperature: float) -> float:
    """SA acceptance probability for a candidate with fitness delta."""
    if delta <= 0:
        return 1.0
    if temperature <= 0:
        return 0.0
    return float(math.exp(-delta / temperature))


def update_velocity(
    velocity: np.ndarray,
    position: np.ndarray,
    personal_best: np.ndarray,
    global_best: np.ndarray,
    w: float,
    c1: float,
    c2: float,
    v_max: float | None = None,
) -> np.ndarray:
    """Update PSO velocity using inertia, cognitive, and social terms."""
    r1 = np.random.rand(*position.shape)
    r2 = np.random.rand(*position.shape)
    new_velocity = (
        w * velocity
        + c1 * r1 * (personal_best - position)
        + c2 * r2 * (global_best - position)
    )
    if v_max is not None:
        new_velocity = np.clip(new_velocity, -v_max, v_max)
    return new_velocity


def update_position(
    position: np.ndarray,
    velocity: np.ndarray,
    demand_min: np.ndarray | None = None,
    demand_max: np.ndarray | None = None,
) -> np.ndarray:
    """Move a PSO particle and optionally clip it to field bounds."""
    new_position = np.array(position, dtype=float) + np.array(velocity, dtype=float)
    if demand_min is None or demand_max is None:
        return new_position
    return clip_to_bounds(new_position, demand_min, demand_max)

