import numpy as np
from dataclasses import dataclass
from typing import Callable

from .iVi import IVIVolterra
from ..kernel.kernel import Kernel


@dataclass
class IVIHawkesProcess:
    kernel: Kernel
    g0_bar: Callable
    rng: np.random.Generator
    g0: Callable = None

    def simulate(self, t_grid, n_paths):
        """


        :param t_grid:
        :param n_paths:
        :return: N, U, lambda as arrays of shape (len(t_grid), n_paths).
        """
        ivi = IVIVolterra(is_continuous=False, kernel=self.kernel, g0_bar=self.g0_bar, rng=self.rng, b=1, c=1, g0=self.g0)
        U, Z, lam = ivi.simulate_u_z_v(t_grid=t_grid, n_paths=n_paths)
        N = Z + U

        # Calculating instantaneous intensity from N
        # K_mat = K.kernel(t_grid[:, None] - t_grid[None, :])
        # K_mat = np.tril(K_mat, k=-1)
        # lam_from_N = g0_const(t_grid).reshape((-1, 1)) + K_mat[:, :-1] @ dN

        # Calculating integrated intensity from N
        K_bar_mat = self.kernel.integrated_kernel(t_grid.reshape(-1, 1) - t_grid[:-1].reshape(1, -1)) - \
                    self.kernel.integrated_kernel(t_grid.reshape(-1, 1) - t_grid[1:].reshape(1, -1))
        K_bar_mat = np.tril(K_bar_mat, k=-1)
        U_from_N = ivi.g0_bar(t_grid).reshape((-1, 1)) + (K_bar_mat @ N[:-1])
        return N, U_from_N, lam
