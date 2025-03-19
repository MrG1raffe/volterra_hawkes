from typing import Callable
import numpy as np


def poisson_field_on_interval(a: float, b: float, intensity: float = 1, rng: np.random.Generator = None):
    if rng is None:
        rng = np.random.default_rng(seed=42)
    number_of_jumps = rng.poisson(lam=intensity * (b - a))
    t_arrivals = a + np.sort(rng.random(size=number_of_jumps)) * (b - a)
    return t_arrivals


def inhomogeneous_poisson_field_on_interval_thinning(a: float, b: float, intensity: Callable, intensity_bound: float,
                                                     rng: np.random.Generator = None):
    if rng is None:
        rng = np.random.default_rng(seed=42)
    t_arrivals = poisson_field_on_interval(a=a, b=b, intensity=intensity_bound, rng=rng)

    unif = rng.random(size=len(t_arrivals))
    t_arrivals_thinned = t_arrivals[unif <= intensity(t_arrivals) / intensity_bound]
    return t_arrivals_thinned


def inhomogeneous_poisson_field_on_interval_inversion(a: float, b: float, integrated_intensity: Callable,
                                                      inv_integrated_intensity: Callable,
                                                      rng: np.random.Generator = None):
    if rng is None:
        rng = np.random.default_rng(seed=42)
    std_poisson_arrivals = poisson_field_on_interval(a=integrated_intensity(a), b=integrated_intensity(b), rng=rng)
    return inv_integrated_intensity(std_poisson_arrivals)