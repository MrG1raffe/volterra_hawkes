import numpy as np
from typing import Callable
from dataclasses import dataclass

from ..kernel.kernel import Kernel


@dataclass
class IVIVolterra:
    is_continuous: bool
    trapeze: bool
    resolvent_IG: bool
    resolvent_alpha: bool
    kernel: Kernel
    g0_bar: Callable
    rng: np.random.Generator
    b: float
    c: float
    g0: Callable = None

    def simulate_u_z(self, t_grid, n_paths):
        n_steps = len(t_grid) - 1
        dt = t_grid[-1] / n_steps


        res = self.kernel.resolvent_as_kernel()

        # Pre-compute certain quantities independent of i and m
        if self.resolvent_IG:
            int = res.integrated_kernel(dt)
            double_int = res.double_integrated_kernel(dt) / dt
        else:
            int = self.kernel.integrated_kernel(dt)
            double_int = self.kernel.double_integrated_kernel(dt) / dt

        # Compute the matrix \bar K_ij
        if self.resolvent_alpha:
            int_matrix = res.integrated_kernel(t_grid[1:].reshape(-1, 1) - t_grid[:-1].reshape(1, -1)) - \
                       res.integrated_kernel(t_grid[:-1].reshape(-1, 1) - t_grid[:-1].reshape(1, -1))
            int_matrix = np.tril(int_matrix, k=-1)
            b_alpha = self.b - 1 #b = 0
        else:
            int_matrix = self.kernel.integrated_kernel(t_grid[1:].reshape(-1, 1) - t_grid[:-1].reshape(1, -1)) - \
                       self.kernel.integrated_kernel(t_grid[:-1].reshape(-1, 1) - t_grid[:-1].reshape(1, -1))
            int_matrix = np.tril(int_matrix, k=-1)
            b_alpha = self.b #b = 1
        

        # Need to stock the Z, U now because of non-markovianity
        dZ, dU = np.zeros((n_steps, n_paths)), np.zeros((n_steps, n_paths))
        g0_bar_diff = np.diff(self.g0_bar(t_grid))

        for i in range(n_steps):
            alpha_i = g0_bar_diff[i] + self.c * int_matrix[i, :] @ dZ + b_alpha * int_matrix[i, :] @ dU
            if self.trapeze:
                mu = alpha_i * int / (2 * double_int * (1 - b_IG * double_int))
                lambda_ = (alpha_i)**2 * int / (4 * (double_int)**3)
                dU_i = self.rng.wald(mean=mu, scale=lambda_, size=n_paths)
                dU_i += alpha_i * (1 - int / (2 * double_int))
                dU_i = np.maximum(0, dU_i)
            else:
                if self.resolvent_IG and not self.resolvent_alpha:
                    alpha_i = alpha_i * (1 + double_int)
                    mu = alpha_i
                    lambda_ = (alpha_i / int)**2    
                    #mu = (1 + int) * alpha_i / (1 + int * (1 - self.b))
                    #lambda_ = ((1 + int) * alpha_i / (self.c * int))**2
                else:
                    mu = alpha_i / (1 - b_alpha * int)
                    lambda_ = (alpha_i / (self.c * int))**2
                    if self.resolvent_alpha:
                        print(np.min(alpha_i))
            dU_i = self.rng.wald(mean=mu, scale=lambda_, size=n_paths)

            #dU_i = self.rng.wald(mean=mu, scale=lambda_, size=n_paths) # inverse_gaussian_sample_vectorized(mu, lambda_, n_paths, self.rng)

            if self.is_continuous:
                dZ_i = ((1 - b_alpha * int) * dU_i - alpha_i) / (self.c * int)
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
