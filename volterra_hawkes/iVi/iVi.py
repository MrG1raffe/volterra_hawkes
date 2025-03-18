import numpy as np
from typing import Callable
from dataclasses import dataclass

from ..kernel.kernel import Kernel


@dataclass
class IVIVolterra:
    is_continuous: bool
    kernel: Kernel
    g0_bar: Callable
    rng: np.random.Generator
    b: float
    c: float
    g0: Callable = None

    def simulate_u_z(self, t_grid, n_paths):
        n_steps = len(t_grid) - 1
        dt = t_grid[-1] / n_steps

        # Pre-compute certain quantities independent of i and m
        K_int_dt = self.kernel.integrated_kernel(dt)

        # Compute the matrix \bar K_ij
        K_bar_matrix = self.kernel.integrated_kernel(t_grid[1:].reshape(-1, 1) - t_grid[:-1].reshape(1, -1)) - \
                       self.kernel.integrated_kernel(t_grid[:-1].reshape(-1, 1) - t_grid[:-1].reshape(1, -1))
        K_bar_matrix = np.tril(K_bar_matrix, k=-1)

        # Need to stock the Z, U now because of non-markovianity
        dZ, dU = np.zeros((n_steps, n_paths)), np.zeros((n_steps, n_paths))
        g0_bar_diff = np.diff(self.g0_bar(t_grid))

        for i in range(n_steps):
            alpha_i = g0_bar_diff[i] + self.c * K_bar_matrix[i, :] @ dZ + self.b * K_bar_matrix[i, :] @ dU
            mu = alpha_i / (1 - self.b * K_int_dt)
            lambda_ = (alpha_i / K_int_dt / self.c) ** 2

            dU_i = self.rng.wald(mean=mu, scale=lambda_, size=n_paths) # inverse_gaussian_sample_vectorized(mu, lambda_, n_paths, self.rng)

            if self.is_continuous:
                dZ_i = ((1 - self.b * K_int_dt) * dU_i - alpha_i) / (self.c * K_int_dt)
            else:
                dZ_i = self.rng.poisson(lam=dU_i, size=n_paths) - dU_i

            dZ[i, :] = dZ_i
            dU[i, :] = dU_i

        Z = np.vstack([np.zeros((1, n_paths)), np.cumsum(dZ, axis=0)])
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
