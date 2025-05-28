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
    resolvent_flag: bool = False
    g0: Callable = None
    g0_bar_res: Callable = None

    def simulate_on_grid(self, t_grid, n_paths):
        """


        :param t_grid:
        :param n_paths:
        :return: N, U, lambda as arrays of shape (len(t_grid), n_paths).
        """

        ivi = IVIVolterra(is_continuous=False, resolvent_flag=self.resolvent_flag, kernel=self.kernel,
                          g0_bar=self.g0_bar, g0_bar_res=self.g0_bar_res, rng=self.rng, b=1, c=1, g0=self.g0)
        U, Z, lam = ivi.simulate_u_z_v(t_grid=t_grid, n_paths=n_paths)
        N = Z + U

        # Calculating integrated intensity from N
        K_bar_mat = self.kernel.integrated_kernel(t_grid.reshape(-1, 1) - t_grid[:-1].reshape(1, -1)) - \
                    self.kernel.integrated_kernel(t_grid.reshape(-1, 1) - t_grid[1:].reshape(1, -1))
        K_bar_mat = np.tril(K_bar_mat, k=-1)
        U_from_N = ivi.g0_bar(t_grid).reshape((-1, 1)) + (K_bar_mat @ N[:-1])
        return N, U, lam

    def simulate_arrivals(self, t_grid, n_paths):
        ivi = IVIVolterra(is_continuous=False, resolvent_flag=self.resolvent_flag, kernel=self.kernel,
                          g0_bar=self.g0_bar, g0_bar_res=self.g0_bar_res, rng=self.rng, b=1, c=1, g0=self.g0)
        U, Z, lam = ivi.simulate_u_z_v(t_grid=t_grid, n_paths=n_paths)
        N = Z + U
        dN = np.round(np.diff(N, axis=0)).astype(int)

        arrivals = []
        for i in range(n_paths):
            uniforms = self.rng.random(size=np.round(N[-1, i]).astype(int))
            jumps = np.repeat(t_grid[:-1], repeats=dN[:, i]) + uniforms * np.repeat(np.diff(t_grid), repeats=dN[:, i])
            arrivals.append(np.sort(jumps))

        return arrivals

    def U_mean(self, t_grid):
        R_mat = np.tril(self.kernel.resolvent(t_grid[:, None] - t_grid[None, :]), k=-1)
        U_mean = self.g0_bar(t_grid) + R_mat @ self.g0_bar(t_grid) * (t_grid[1] - t_grid[0])
        return U_mean

    def lam_from_jumps(self, t, t_jumps):
        return self.g0(t) + np.sum(np.where(t.reshape((-1, 1)) > t_jumps.reshape((1, -1)),
                                            self.kernel(t.reshape((-1, 1)) - t_jumps.reshape((1, -1))),
                                            0), axis=1)

    def U_from_jumps(self, t, t_jumps):
        return self.g0_bar(t) + np.sum(np.where(t.reshape((-1, 1)) > t_jumps.reshape((1, -1)),
                                                self.kernel.integrated_kernel(t.reshape((-1, 1)) - t_jumps.reshape((1, -1))),
                                                0), axis=1)

    @staticmethod
    def N_from_jumps(t, t_jumps):
        return np.sum(t.reshape((-1, 1)) >= t_jumps.reshape((1, -1)), axis=1)
