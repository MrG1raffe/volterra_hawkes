import numpy as np
from numpy.typing import NDArray
from numpy import float64
from typing import Callable, Union

from .poisson import inhomogeneous_poisson_field_on_interval_thinning, inhomogeneous_poisson_field_on_interval_inversion
from ..kernel.kernel import Kernel


def simulate_hawkes(
    T: float,
    g0: Callable,
    g0_upper_bound: float,
    kernel: Kernel,
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
                                                                                                 integrated_intensity=kernel.integrated_kernel,
                                                                                                 inv_integrated_intensity=kernel.inv_integrated_kernel,
                                                                                                 rng=rng)
        # print("Parent:", parent_arrival, "Number of descendents:", len(descendent_arrials), "Descendents:", descendent_arrials)
        to_concatenate.append(descendant_arrivals)
        if ptr == len(hawkes_arrivals) - 1 and to_concatenate:
            hawkes_arrivals = np.concatenate([hawkes_arrivals] + to_concatenate)
            to_concatenate = []
        ptr += 1

    hawkes_arrivals = np.sort(hawkes_arrivals)
    return hawkes_arrivals


def simulate_hawkes_ogata(
    T: float,
    mu: float,
    kernel: Union[Callable, Kernel],
    rng: np.random.Generator = None,
    eps: float = 1e-11,
    batch_size: int = 100
):
    if rng is None:
        rng = np.random.default_rng(seed=42)

    ptr = 0
    batch_iter = 0
    event_counter = 0

    uniform_batch = rng.uniform(size=(batch_size, 2))
    arrivals = np.empty(batch_size)

    while ptr <= T:
        M = mu + kernel(ptr - arrivals[:event_counter] + eps).sum()
        arrival_cand = ptr - np.log(uniform_batch[batch_iter, 0]) / M
        if uniform_batch[batch_iter, 1] < (mu + kernel(arrival_cand - arrivals[:event_counter]).sum()) / M :
            arrivals[event_counter] = arrival_cand
            event_counter += 1
            if event_counter == arrivals.size:
                arrivals = np.concatenate([arrivals, np.empty(batch_size)])

        ptr = arrival_cand

        batch_iter += 1
        if batch_iter == batch_size:
            uniform_batch = rng.uniform(size=(batch_size, 2))
            batch_iter = 0

    #  print("Number of jumps: ", len(arrivals), "Number of iterations: ", number_of_iter, "Acceptance rate: ", len(arrivals) / number_of_iter)
    return arrivals[:event_counter - 1]


def lam_from_jumps(t, t_jumps, kernel: Kernel, g0: Callable):
    return g0(t) + np.sum(np.where(t.reshape((-1, 1)) > t_jumps.reshape((1, -1)),
                                   kernel(t.reshape((-1, 1)) - t_jumps.reshape((1, -1))),
                                   0), axis=1)


def U_from_jumps(t, t_jumps, kernel: Kernel, g0_bar: Callable):
    return g0_bar(t) + np.sum(np.where(t.reshape((-1, 1)) > t_jumps.reshape((1, -1)),
                                       kernel.integrated_kernel(t.reshape((-1, 1)) - t_jumps.reshape((1, -1))),
                                       0), axis=1)


def N_from_jumps(t, t_jumps):
    return np.sum(t.reshape((-1, 1)) >= t_jumps.reshape((1, -1)), axis=1)
