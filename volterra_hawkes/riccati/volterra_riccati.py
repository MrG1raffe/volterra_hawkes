import numpy as np
from typing import Callable, Tuple

from ..kernel.kernel import Kernel


def right_point_adams_scheme(
        T: float,
        n_steps: int,
        K: Kernel,
        F: Callable
    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve a Volterra-type integral equation using the right-point Adams scheme.

    This method approximates the solution of an integral equation of the form:

        ψ(t) = ∫₀^t K(t - s) F(s, ψ(s)) ds

    using a right-point quadrature (Adams) scheme on a uniform grid.

    Parameters
    ----------
    T : float
        End time of the integration interval [0, T].
    n_steps : int
        Number of time steps in the uniform discretization.
    K : Kernel
        Kernel function of the Volterra integral equation.
    F : Callable
        Function F(t, ψ) representing the integrand of the Volterra equation.

    Returns
    -------
    psi : np.ndarray
        Array of approximated values of ψ at the grid points.
    F_arr : np.ndarray
        Array of F(t_i, ψ_i) evaluated at the grid points.

    Notes
    -----
    - The scheme uses right-point evaluation for the integral approximation.
    - `t_grid` is uniformly spaced from 0 to T with `n_steps + 1` points.
    - `delta_K_bar` represents the increments of the integrated kernel at the grid points.
    """
    t_grid = np.linspace(0, T, n_steps + 1)
    psi = np.zeros_like(t_grid)
    F_arr = np.zeros_like(t_grid)
    F_arr[0] = F(t_grid[0], psi[0])

    delta_K_bar = np.diff(K.integrated_kernel(t_grid))
    for i in range(n_steps):
        F_arr[i + 1] = F_arr[i]
        psi_pred = F_arr[1:i+2] @ np.flip(delta_K_bar[:i+1])
        F_arr[i + 1] = F(t_grid[i + 1], psi_pred)
        psi[i + 1] = F_arr[1:i+2] @ np.flip(delta_K_bar[:i+1])  # psi_pred + (F_arr[i + 1] - F_arr[i]) * delta_K_bar[i]
        F_arr[i + 1] = F(t_grid[i + 1], psi[i + 1])
    return psi, F_arr
