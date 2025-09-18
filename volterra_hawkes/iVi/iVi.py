import warnings

import numpy as np
from typing import Callable
from dataclasses import dataclass

from ..kernel.kernel import Kernel
from ..kernel.exponential_kernel import ExponentialKernel


@dataclass
class IVIVolterra:
    is_continuous: bool
    resolvent_flag: bool
    kernel: Kernel
    g0_bar: Callable
    rng: np.random.Generator
    b: float
    c: float
    g0: Callable = None
    g0_bar_res: Callable = None

    def simulate_u_z(self, t_grid, n_paths):
        n_steps = len(t_grid) - 1
        dt = t_grid[-1] / n_steps

        # Compute the matrix \bar K_ij
        if self.resolvent_flag:
            ivi_kernel = self.kernel.resolvent
            b_alpha = self.b - 1  # b = 0
            g0_bar_diff = np.diff(self.g0_bar_res(t_grid))
        else:
            ivi_kernel = self.kernel
            b_alpha = self.b  # b = 1
            g0_bar_diff = np.diff(self.g0_bar(t_grid))

        int_kernel = ivi_kernel.integrated_kernel
        int_ker_dt = int_kernel(dt)
        if (1 - b_alpha * int_ker_dt) < 0:
            raise ValueError("The denominator (1 - b * K_bar) cannot be negative. Reduce the discretization step.")

        markov_flag = isinstance(ivi_kernel, ExponentialKernel)

        if not markov_flag:
            int_matrix = int_kernel(t_grid[1:].reshape(-1, 1) - t_grid[:-1].reshape(1, -1)) - \
                         int_kernel(t_grid[:-1].reshape(-1, 1) - t_grid[:-1].reshape(1, -1))  # k_1
            int_matrix = np.tril(int_matrix, k=-1)
        else:
            mult_coef = np.exp(-ivi_kernel.lam * dt)
            k = (int_kernel(2*dt) - int_kernel(dt)) # k = k1

        dZ, dU, d_xi = np.zeros((n_steps, n_paths)), np.zeros((n_steps, n_paths)), np.zeros((n_steps, n_paths))

        alpha_i = 0
        for i in range(n_steps):
            if not markov_flag:
                alpha_i = g0_bar_diff[i] + self.c * int_matrix[i, :] @ dZ + b_alpha * int_matrix[i, :] @ d_xi
            else:
                if i == 0:
                    alpha_i = g0_bar_diff[i]
                else:
                    alpha_i = (g0_bar_diff[i] + (alpha_i - g0_bar_diff[i - 1]) * mult_coef +
                               (self.c * dZ[i - 1, :] + b_alpha * dU[i - 1, :]) * k)
            if np.any(alpha_i < 0):
                scheme_name = "iVi" if not self.resolvent_flag else "iVi Res"
                warnings.warn(f"Negative alpha encountered in {scheme_name} scheme. Setting to 0.")
                alpha_i = np.maximum(alpha_i, 1e-6)

            mu = alpha_i / (1 - b_alpha * int_ker_dt)
            lambda_ = (alpha_i / (self.c * int_ker_dt))**2
            d_xi_i = self.rng.wald(mean=mu, scale=lambda_, size=n_paths)

            if self.is_continuous:
                dU_i = d_xi_i
                dZ_i = ((1 - b_alpha * int_ker_dt) * dU_i - alpha_i) / (self.c * int_ker_dt)
            else:
                dN_i = self.rng.poisson(lam=d_xi_i, size=n_paths)
                dZ_i = dN_i - d_xi_i
                dU_i = (alpha_i + int_ker_dt * self.c * dN_i) / (1 + int_ker_dt * (self.c - b_alpha))

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
        if self.g0 is None:
            raise ValueError("g0 should be specified to simulate V.")

        U, Z = self.simulate_u_z(t_grid=t_grid, n_paths=n_paths)
        dU, dZ = np.diff(U, axis=0), np.diff(Z, axis=0)

        K_mat = self.kernel(t_grid[:, None] - t_grid[None, :])
        K_mat = np.tril(K_mat, k=-1)
        V = self.g0(t_grid).reshape((-1, 1)) + self.c * K_mat[:, :-1] @ dZ + self.b * K_mat[:, :-1] @ dU
        return U, Z, V
