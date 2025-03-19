import numpy as np
from numpy.typing import NDArray
from numpy import float64
from typing import Callable

from .poisson import inhomogeneous_poisson_field_on_interval_thinning, inhomogeneous_poisson_field_on_interval_inversion


def simulate_hawkes(
    T: float,
    g0: Callable,
    g0_upper_bound: float,
    integrated_kernel: Callable,
    inv_integrated_kernel: Callable,
    rng: np.random.Generator = None
) -> NDArray[float64]:
    if rng is None:
        rng = np.random.default_rng(seed=42)
    # simulate the jumps of an inhomogeneous Poisson with intensity g_0(t) via thinning
    hawkes_arrivals = inhomogeneous_poisson_field_on_interval_thinning(a=0, b=T, intensity=g0,
                                                                       intensity_bound=g0_upper_bound, rng=rng)

    ptr = 0
    to_concatenate = []
    while ptr < len(hawkes_arrivals):
        parent_arrival = hawkes_arrivals[ptr]
        # simulating inhomogeneous poisson on [parent_arrival, T] with intensity K(t - parent_arrival)
        descendant_arrivals = parent_arrival + inhomogeneous_poisson_field_on_interval_inversion(a=0, b=T - parent_arrival,
                                                                                                integrated_intensity=integrated_kernel,
                                                                                                inv_integrated_intensity=inv_integrated_kernel,
                                                                                                rng=rng)
        # print("Parent:", parent_arrival, "Number of descendents:", len(descendent_arrials), "Descendents:", descendent_arrials)
        to_concatenate.append(descendant_arrivals)
        if ptr == len(hawkes_arrivals) - 1 and to_concatenate:
            hawkes_arrivals = np.concatenate([hawkes_arrivals] + to_concatenate)
            to_concatenate = []
        ptr += 1

    hawkes_arrivals = np.sort(hawkes_arrivals)
    return hawkes_arrivals
