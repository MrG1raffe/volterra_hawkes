import numpy as np
from dataclasses import dataclass
from dataclasses import field

from .iVi import IVIVolterra


@dataclass
class IVIVolterraVolModel(IVIVolterra):
    rho: float = 0

    def simulate_price(self, t_grid, n_paths, S0: float = 1):
        """
        :param t_grid:
        :param n_paths:
        :param S0:
        :return: three arrays S, U, Z, V of shape (len(t_grid), n_paths).
        """
        U, Z, V = self.simulate_u_z_v(n_paths=n_paths, t_grid=t_grid)
        normal_ort = self.rng.normal(size=(len(t_grid), n_paths))
        logS = -0.5 * U + self.rho * Z + np.sqrt(1 - self.rho**2) * \
               np.cumsum(np.sqrt(np.diff(U, axis=0, prepend=U[0:1])) * normal_ort, axis=0)
        return S0 * np.exp(logS), U, Z, V
