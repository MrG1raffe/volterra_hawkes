from typing import Callable
import numpy as np


def poisson_field_on_interval(
        a: float,
        b: float,
        intensity: float = 1,
        rng: np.random.Generator = None
    ) -> np.ndarray:
    """
    Simulate a homogeneous Poisson process on a time interval [a, b].

    The number of events is sampled from a Poisson distribution, and event
    times are uniformly distributed over the interval.

    Parameters
    ----------
    a : float
        Start time of the interval.
    b : float
        End time of the interval.
    intensity : float, optional
        Constant rate (λ) of the Poisson process. Default is 1.
    rng : np.random.Generator, optional
        Random number generator for reproducibility. Default is a new generator with seed 42.

    Returns
    -------
    t_arrivals : np.ndarray
        Sorted array of event times in [a, b].
    """
    if rng is None:
        rng = np.random.default_rng(seed=42)
    number_of_jumps = rng.poisson(lam=intensity * (b - a))
    t_arrivals = a + np.sort(rng.random(size=number_of_jumps)) * (b - a)
    return t_arrivals


def inhomogeneous_poisson_field_on_interval_thinning(
        a: float,
        b: float,
        intensity: Callable,
        intensity_bound: float,
        rng: np.random.Generator = None
    ) -> np.ndarray:
    """
    Simulate an inhomogeneous Poisson process on [a, b] using the thinning algorithm.

    A homogeneous Poisson process with rate `intensity_bound` is generated first,
    and then points are thinned according to the target intensity function.

    Parameters
    ----------
    a : float
       Start time of the interval.
    b : float
       End time of the interval.
    intensity : Callable
       Time-dependent intensity function λ(t).
    intensity_bound : float
       Upper bound of the intensity function for thinning.
    rng : np.random.Generator, optional
       Random number generator for reproducibility. Default is a new generator with seed 42.

    Returns
    -------
    t_arrivals_thinned : np.ndarray
       Array of event times after thinning.
    """
    if rng is None:
        rng = np.random.default_rng(seed=42)
    t_arrivals = poisson_field_on_interval(a=a, b=b, intensity=intensity_bound, rng=rng)
    unif = rng.random(size=len(t_arrivals))
    t_arrivals_thinned = t_arrivals[unif <= intensity(t_arrivals) / intensity_bound]
    return t_arrivals_thinned


def inhomogeneous_poisson_field_on_interval_inversion(
        a: float,
        b: float,
        integrated_intensity: Callable,
        inv_integrated_intensity: Callable,
        rng: np.random.Generator = None
    ) -> np.ndarray:
    """
    Simulate an inhomogeneous Poisson process on [a, b] using the inversion method.

    Events are simulated in the standard Poisson space of the integrated intensity
    and then mapped back using the inverse of the integrated intensity.

    Parameters
    ----------
    a : float
        Start time of the interval.
    b : float
        End time of the interval.
    integrated_intensity : Callable
        Integrated intensity function Λ(t) = ∫₀^t λ(s) ds.
    inv_integrated_intensity : Callable
        Inverse function of the integrated intensity Λ⁻¹.
    rng : np.random.Generator, optional
        Random number generator for reproducibility. Default is a new generator with seed 42.

    Returns
    -------
    t_arrivals : np.ndarray
        Array of event times in [a, b].
    """
    if rng is None:
        rng = np.random.default_rng(seed=42)
    std_poisson_arrivals = poisson_field_on_interval(a=integrated_intensity(a), b=integrated_intensity(b), rng=rng)
    return inv_integrated_intensity(std_poisson_arrivals)
