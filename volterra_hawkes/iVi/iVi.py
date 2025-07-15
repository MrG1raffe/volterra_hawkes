import warnings

import numpy as np
from typing import Callable
from dataclasses import dataclass

from ..kernel.kernel import Kernel


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

    def simulate_u_z(self, t_grid, n_paths, return_alpha: bool = False):
        n_steps = len(t_grid) - 1
        dt = t_grid[-1] / n_steps
        res = self.kernel.resolvent

        # Compute the matrix \bar K_ij
        if self.resolvent_flag:
            int_ker = res.integrated_kernel(dt)
            int_matrix = res.integrated_kernel(t_grid[1:].reshape(-1, 1) - t_grid[:-1].reshape(1, -1)) - \
                         res.integrated_kernel(t_grid[:-1].reshape(-1, 1) - t_grid[:-1].reshape(1, -1))
            int_matrix = np.tril(int_matrix, k=-1)
            b_alpha = self.b - 1  # b = 0
        else:
            int_ker = self.kernel.integrated_kernel(dt)
            int_matrix = self.kernel.integrated_kernel(t_grid[1:].reshape(-1, 1) - t_grid[:-1].reshape(1, -1)) - \
                         self.kernel.integrated_kernel(t_grid[:-1].reshape(-1, 1) - t_grid[:-1].reshape(1, -1))
            int_matrix = np.tril(int_matrix, k=-1)
            b_alpha = self.b  # b = 1

        if (1 - b_alpha * int_ker) < 0:
            raise ValueError("The denominator (1 - b * K_bar) cannot be negative. Reduce the discretization step.")

        # Need to stock the Z, U now because of non-markovianity
        dZ, dU, d_alpha = np.zeros((n_steps, n_paths)), np.zeros((n_steps, n_paths)), np.zeros((n_steps, n_paths))
        if self.resolvent_flag:
            g0_bar_diff = np.diff(self.g0_bar_res(t_grid))
        else:
            g0_bar_diff = np.diff(self.g0_bar(t_grid))

        # g0_bar_no_res_diff = np.diff(self.g0_bar(t_grid))

        for i in range(n_steps):
            alpha_i = g0_bar_diff[i] + self.c * int_matrix[i, :] @ dZ + b_alpha * int_matrix[i, :] @ dU

            if np.any(alpha_i < 0):
                scheme_name = "iVi" if not self.resolvent_flag else "iVi Res"
                warnings.warn(f"Negative alpha encountered in {scheme_name} scheme. Setting to 0.")
                alpha_i = np.maximum(alpha_i, 1e-6)
            mu = alpha_i / (1 - b_alpha * int_ker)
            # print(mu)
            lambda_ = (alpha_i / (self.c * int_ker))**2
            dU_i = self.rng.wald(mean=mu, scale=lambda_, size=n_paths)

            if self.is_continuous:
                dZ_i = ((1 - b_alpha * int_ker) * dU_i - alpha_i) / (self.c * int_ker)
            else:
                dZ_i = self.rng.poisson(lam=dU_i, size=n_paths) - dU_i

            dZ[i, :] = dZ_i
            d_alpha[i, :] = alpha_i
            dU[i, :] = dU_i

        # d_alpha = np.maximum(d_alpha, g0_bar_no_res_diff[:, None])

        Z = np.vstack([np.zeros((1, n_paths)), np.cumsum(dZ, axis=0)])
        U = np.vstack([np.zeros((1, n_paths)), np.cumsum(dU, axis=0)])
        alpha = np.vstack([np.zeros((1, n_paths)), np.cumsum(d_alpha, axis=0)])

        # alpha = np.maximum(alpha, self.g0_bar(t_grid)[:, None])

        if return_alpha:
            return U, Z, alpha
        else:
            return U, Z

    def simulate_u_z_v(self, t_grid, n_paths, return_alpha: bool = False):
        if self.g0 is None:
            raise ValueError("g0 should be specified to simulate V.")

        U, Z, alpha = self.simulate_u_z(t_grid=t_grid, n_paths=n_paths, return_alpha=True)
        dU, dZ = np.diff(U, axis=0), np.diff(Z, axis=0)

        K_mat = self.kernel(t_grid[:, None] - t_grid[None, :])
        K_mat = np.tril(K_mat, k=-1)
        V = self.g0(t_grid).reshape((-1, 1)) + self.c * K_mat[:, :-1] @ dZ + self.b * K_mat[:, :-1] @ dU
        if return_alpha:
            return U, Z, V, alpha
        else:
            return U, Z, V
