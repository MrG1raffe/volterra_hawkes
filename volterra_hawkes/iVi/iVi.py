import warnings

import numpy as np
from typing import Callable
from dataclasses import dataclass
from copy import deepcopy

from ..kernel.constant_kernel import ConstantKernel
from ..kernel.kernel import Kernel
from ..kernel.exponential_kernel import ExponentialKernel


@dataclass
class IVIVolterra:
    """
    Simulator for the (iVi) and (iVi Res) Volterra schemes.

    This class implements simulation methods for interacting
    Volterra-type processes, with support for both the standard
    kernel formulation and its resolvent version. It can produce
    trajectories of the auxiliary processes (U, Z), and optionally
    the instantaneous variance process (V).

    Integrated variance dynamics
    ---------------------------
    The process :math:`U_t` follows the Volterra-type stochastic integral equation:

    .. math::

        U_t = \int_0^t g_0(s)\,ds + \int_0^t K(t-s) ( b\,U_s + c\,Z_s)\,ds,

    where :math:`K` is the kernel function, :math:`g_0(t)` is the input function,
    and :math:`Z_t` is a martingale auxiliary process.


    Attributes
    ----------
    is_continuous : bool
        If True, simulates the continuous version of the scheme.
        If False, uses the jump–Hawkes version.
    resolvent_flag : bool
        If True, uses the resolvent kernel and `g0_bar_res` for the scheme.
        Otherwise, uses the standard kernel and `g0_bar`.
    kernel : Kernel
        The kernel function K(t), must implement `__call__` and
        `integrated_kernel`.
    g0_bar : Callable
        Function ḡ₀(t) defining the integrated input function.
        Required if `resolvent_flag` is False.
    rng : numpy.random.Generator
        Random number generator used for sampling (Poisson/Wald).
    b : float
        Coefficient `b` in the scheme. Common choices: 0 (resolvent case),
        or 1 (standard case).
    c : float
        Coefficient `c` in the scheme.
    g0_bar_res : Callable, optional
        Function ḡ₀ʳ(t) = ḡ₀ + R ★ ḡ₀, where R is the resolvent of bK, required if `resolvent_flag` is True.
    g0 : Callable, optional
        Function g₀(t), used when simulating the instantaneous
        variance process V.

    Notes
    -----
    - If `resolvent_flag` is set, then `g0_bar_res` must be provided.
    - If `resolvent_flag` is False, then `g0_bar` must be provided.
    - The scheme uses warnings to notify when negative `alpha` values
      are encountered in the iVi Res (these are clipped to a small positive number).
    """
    is_continuous: bool
    resolvent_flag: bool
    kernel: Kernel
    g0_bar: Callable
    rng: np.random.Generator
    b: float
    c: float
    g0_bar_res: Callable = None  # Optional; should be specified to use the resolvent version of the scheme
    g0: Callable = None          # Optional; should be specified if one wants to simulate the instantaneous variance process


    def simulate_u_z(self, t_grid, n_paths):
        """
        Simulate the processes U and Z.

        Parameters
        ----------
        t_grid : array_like of shape (n_steps + 1,)
            Increasing time grid for the simulation.
        n_paths : int
            Number of independent sample paths to simulate.

        Returns
        -------
        U : ndarray of shape (n_steps + 1, n_paths)
            Simulated process U(t).
        Z : ndarray of shape (n_steps + 1, n_paths)
            Simulated process Z(t).

        Raises
        ------
        ValueError
            If the stability condition `(1 - b * K̄(dt)) < 0` is violated,
            which typically indicates that the time step is too large.

        Warns
        -----
        UserWarning
            If negative `alpha` values are encountered during simulation,
            they are clipped to small positive values.
        """
        n_steps = len(t_grid) - 1
        dt = t_grid[-1] / n_steps

        # Compute the matrix \bar K_ij
        if self.resolvent_flag:
            b_kernel = deepcopy(self.kernel)
            b_kernel.c = self.kernel.c * self.b
            ivi_kernel = b_kernel.resolvent  # resolvent of the kernel bK
            b_scheme = 0
            c_scheme = self.c / self.b
            g0_bar_diff = np.diff(self.g0_bar_res(t_grid))
        else:
            ivi_kernel = self.kernel
            b_scheme = self.b
            c_scheme = self.c
            g0_bar_diff = np.diff(self.g0_bar(t_grid))

        int_kernel = ivi_kernel.integrated_kernel
        int_ker_dt = int_kernel(dt)
        if (1 - b_scheme * int_ker_dt) < 0:
            raise ValueError("The denominator (1 - b * K_bar) cannot be negative. Reduce the discretization step.")

        if isinstance(ivi_kernel, ConstantKernel) or isinstance(ivi_kernel, ExponentialKernel):
            # Markovian update of alpha.
            if isinstance(ivi_kernel, ConstantKernel):
                mult_coef = 1
            else:
                mult_coef = np.exp(-ivi_kernel.lam * dt)
            k = (int_kernel(2 * dt) - int_kernel(dt))  # k = k1

            def update_alpha(alpha_prev, g0_bar_diff, dZ, dU, d_xi):
                if i == 0:
                    alpha_i = g0_bar_diff[i]
                else:
                    alpha_i = (g0_bar_diff[i] + (alpha_prev - g0_bar_diff[i - 1]) * mult_coef +
                               (c_scheme * dZ[i - 1, :] + b_scheme * dU[i - 1, :]) * k)
                return alpha_i
        else:
            # General non-Markovian update
            int_matrix = int_kernel(t_grid[1:].reshape(-1, 1) - t_grid[:-1].reshape(1, -1)) - \
                         int_kernel(t_grid[:-1].reshape(-1, 1) - t_grid[:-1].reshape(1, -1))  # k_1
            int_matrix = np.tril(int_matrix, k=-1)

            def update_alpha(alpha_prev, g0_bar_diff, dZ, dU, d_xi):
                return g0_bar_diff[i] + c_scheme * int_matrix[i, :] @ dZ + b_scheme * int_matrix[i, :] @ d_xi

        dZ, dU, d_xi = np.zeros((n_steps, n_paths)), np.zeros((n_steps, n_paths)), np.zeros((n_steps, n_paths))

        alpha_i = 0
        for i in range(n_steps):
            alpha_i = update_alpha(alpha_i, g0_bar_diff, dZ, dU, d_xi)
            if np.any(alpha_i < 0):
                scheme_name = "iVi" if not self.resolvent_flag else "iVi Res"
                warnings.warn(f"Negative alpha encountered in {scheme_name} scheme. Setting to 0.")
                alpha_i = np.maximum(alpha_i, 1e-6)

            mu = alpha_i / (1 - b_scheme * int_ker_dt)
            lambda_ = (alpha_i / (c_scheme * int_ker_dt))**2
            d_xi_i = self.rng.wald(mean=mu, scale=lambda_, size=n_paths)

            if self.is_continuous:
                dU_i = d_xi_i
                dZ_i = ((1 - b_scheme * int_ker_dt) * dU_i - alpha_i) / (c_scheme * int_ker_dt)
            else:
                dN_i = self.rng.poisson(lam=d_xi_i, size=n_paths)
                dZ_i = dN_i - d_xi_i
                dU_i = (alpha_i + int_ker_dt * c_scheme * dN_i) / (1 + int_ker_dt * (c_scheme - b_scheme))

            d_xi[i, :] = d_xi_i
            dZ[i, :] = dZ_i
            dU[i, :] = dU_i

        if self.is_continuous:
            Z = np.vstack([np.zeros((1, n_paths)), np.cumsum(dZ, axis=0)])
        else:
            Z = np.vstack([np.zeros((1, n_paths)), np.cumsum(dZ + d_xi - dU, axis=0)]) # = dN - dU
        U = np.vstack([np.zeros((1, n_paths)), np.cumsum(dU, axis=0)])

        return U, Z

    def simulate_u_z_v(self, t_grid, n_paths):
        """
        Simulate the processes U, Z, and V.

        This method extends `simulate_u_z` by also computing the
        instantaneous variance process V, if the function `g0`
        was provided at initialization.

        Parameters
        ----------
        t_grid : array_like of shape (n_steps + 1,)
            Increasing time grid for the simulation.
        n_paths : int
            Number of independent sample paths to simulate.

        Returns
        -------
        U : ndarray of shape (n_steps + 1, n_paths)
            Simulated process U(t).
        Z : ndarray of shape (n_steps + 1, n_paths)
            Simulated process Z(t).
        V : ndarray of shape (n_steps + 1, n_paths)
            Instantaneous variance process V(t).
            If `g0` is not specified, returns an array filled with NaN.
        """
        U, Z = self.simulate_u_z(t_grid=t_grid, n_paths=n_paths)
        dU, dZ = np.diff(U, axis=0), np.diff(Z, axis=0)

        K_mat = self.kernel(t_grid[:, None] - t_grid[None, :])
        K_mat = np.tril(K_mat, k=-1)

        if self.g0 is None:
            V = np.zeros_like(Z) * np.nan
        else:
            V = self.g0(t_grid).reshape((-1, 1)) + self.c * K_mat[:, :-1] @ dZ + self.b * K_mat[:, :-1] @ dU

        return U, Z, V
