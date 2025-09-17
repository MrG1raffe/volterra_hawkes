import numpy as np
from numpy.typing import NDArray
from numpy import float64
from typing import Callable, Union

from .poisson import inhomogeneous_poisson_field_on_interval_thinning, inhomogeneous_poisson_field_on_interval_inversion
from ..kernel.exponential_kernel import ExponentialKernel
from ..kernel.kernel import Kernel
from ..utility.sim_counter import SimCounter


def simulate_hawkes(
    T: float,
    g0: Callable,
    g0_upper_bound: float,
    kernel: Kernel,
    rng: np.random.Generator = None,
    sim_counter: SimCounter = None
) -> NDArray[float64]:
    if rng is None:
        rng = np.random.default_rng(seed=42)
    # simulate the jumps of an inhomogeneous Poisson with intensity g_0(t) via thinning
    hawkes_arrivals = inhomogeneous_poisson_field_on_interval_thinning(a=0, b=T, intensity=g0,
                                                                                   intensity_bound=g0_upper_bound,
                                                                                   rng=rng, sim_counter=sim_counter)

    ptr = 0
    to_concatenate = []
    while ptr < len(hawkes_arrivals):
        parent_arrival = hawkes_arrivals[ptr]
        # simulating inhomogeneous poisson on [parent_arrival, T] with intensity K(t - parent_arrival)
        descendant_arrivals = parent_arrival + inhomogeneous_poisson_field_on_interval_inversion(a=0, b=T - parent_arrival,
                                                                                                 integrated_intensity=kernel.integrated_kernel,
                                                                                                 inv_integrated_intensity=kernel.inv_integrated_kernel,
                                                                                                 rng=rng, sim_counter=sim_counter)
        # print("Parent:", parent_arrival, "Number of descendents:", len(descendent_arrials), "Descendents:", descendent_arrials)
        to_concatenate.append(descendant_arrivals)
        if ptr == len(hawkes_arrivals) - 1 and to_concatenate:
            hawkes_arrivals = np.concatenate([hawkes_arrivals] + to_concatenate)
            to_concatenate = []
        ptr += 1

    hawkes_arrivals = np.sort(hawkes_arrivals)
    return hawkes_arrivals

def simulate_exponential_hawkes(
    T: float,
    kernel: ExponentialKernel,
    mean_intensity: float,
    init_intensity: float,
    rng: np.random.Generator = None,
    sim_counter: SimCounter = None
) -> NDArray[float64]:
    """
    Simulates a trajectory of an exponential Hawkes process with intensity

    λ_t = a + (λ_0 - a)exp(-lam * t) + sum_{T_k < t} c exp(-lam(t - T_k)).

    :param T:
    :param kernel:
    :param mean_intensity:
    :param init_intensity:
    :param rng:
    :param sim_counter:
    :return:
    """
    if rng is None:
        rng = np.random.default_rng(seed=42)

    hawkes_arrivals = []
    lam, c = kernel.lam, kernel.c

    current_intensity = init_intensity + 0.00001
    current_arrival = 0

    while current_arrival < T:
        U = rng.uniform(size=2)
        if sim_counter is not None:
            sim_counter.add(U.size)
        D = 1 + lam * np.log(U[0]) / (current_intensity - mean_intensity)
        S_2 = -np.log(U[1]) / mean_intensity
        if D > 0:
            S_1 = -np.log(D) / lam
            S = min(S_1, S_2)
        else:
            S = S_2
        current_arrival += S
        current_intensity = (current_intensity - mean_intensity) * np.exp(-lam * S) + mean_intensity + c
        hawkes_arrivals.append(current_arrival)

    return np.array(hawkes_arrivals)[:-1]  # the last arrival is > T.


def simulate_hawkes_ogata(
    T: float,
    mu: float,
    kernel: Union[Callable, Kernel],
    rng: np.random.Generator = None,
    eps: float = 1e-11,
    batch_size: int = 10,
    sim_counter: SimCounter = None,
    decreasing_kernel: Union[Callable, Kernel] = None # used to get correct bounds for non-monotone kernels
):
    if rng is None:
        rng = np.random.default_rng(seed=42)
    if decreasing_kernel is None:
        decreasing_kernel = kernel

    ptr = 0
    batch_iter = 0
    event_counter = 0

    uniform_batch = rng.uniform(size=(batch_size, 2))
    if sim_counter is not None:
        sim_counter.add(uniform_batch.size)
    arrivals = np.empty(batch_size)

    thinning_step = 0
    M = mu
    is_accepted = False
    while ptr < T:
        if isinstance(kernel, ExponentialKernel):
            M = mu + np.exp(-thinning_step * kernel.lam) * (M - mu) + is_accepted * kernel.c
        else:
            M = mu + decreasing_kernel(ptr - arrivals[:event_counter] + eps).sum()
        thinning_step = - np.log(uniform_batch[batch_iter, 0]) / M
        arrival_cand = ptr + thinning_step

        is_accepted = uniform_batch[batch_iter, 1] < (mu + kernel(arrival_cand - arrivals[:event_counter]).sum()) / M
        if is_accepted:
            arrivals[event_counter] = arrival_cand
            event_counter += 1
            if event_counter == arrivals.size:
                arrivals = np.concatenate([arrivals, np.empty(batch_size)])

        ptr = arrival_cand
        batch_iter += 1
        if batch_iter == batch_size:
            uniform_batch = rng.uniform(size=(batch_size, 2))
            if sim_counter is not None:
                sim_counter.add(uniform_batch.size)
            batch_iter = 0

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
