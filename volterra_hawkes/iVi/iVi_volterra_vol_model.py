import numpy as np
from dataclasses import dataclass

from .iVi import IVIVolterra


@dataclass
class IVIVolterraVolModel(IVIVolterra):
    r"""
    Inhomogeneous Volterra Stochastic Volatility Model with correlated noise.

    This class extends the base :class:`IVIVolterra` model by including a
    correlation parameter `rho` that links the Brownian motion driving the
    volatility process with the Brownian motion driving the stock price.

    The integrated variance process :math:`U_{t} = \int_0^t V_s\, ds`, satisfies the Volterra-type
    integral equation

    .. math::

        U_{t} = \int_0^t g_0(s)\,ds
                  + \int_0^t K(t-s) \big( b\,U_{s} + c\,Z_{s} \big)\,ds, \\

    where :math:`K` is the Volterra kernel, :math:`g_0` is the input function
    defining the initial variance curve, and :math:`Z` is the subordinated process.

    The stock price :math:`S_t` follows martingale dynamics driven by a Brownian motion :math:`B_{t}` correlated with
    :math:`Z_t` and :math:`U_{t}` as integrated variance.

    Attributes
    ----------
    rho : float, default=0
        Correlation between the Brownian motion driving the asset price
        and the Brownian motion driving the variance process.
    """
    rho: float = 0

    def simulate_price(self, t_grid, n_paths, S0: float = 1):
        """
        Simulate asset price paths together with variance-related processes.

        The method generates Monte Carlo paths for:
        - The stock price `S`
        - The integrated variance `U`
        - The subordinated process `Z`
        - The instantaneous variance `V`

        The dynamics of the log-price include a correlation term `rho` that
        couples the asset and variance innovations.

        Parameters
        ----------
        t_grid : array_like
            Discrete time grid of shape (n_steps,), including the initial time.
        n_paths : int
            Number of simulated Monte Carlo paths.
        S0 : float, optional, default=1
            Initial stock price.

        Returns
        -------
        S : ndarray
            Simulated stock price paths of shape (len(t_grid), n_paths).
        U : ndarray
            Integrated variance process of shape (len(t_grid), n_paths).
        Z : ndarray
            Subordinated process of shape (len(t_grid), n_paths).
        V : ndarray
            Instantaneous variance of shape (len(t_grid), n_paths).
        """
        U, Z, V = self.simulate_u_z_v(n_paths=n_paths, t_grid=t_grid)
        normal_ort = self.rng.normal(size=(len(t_grid), n_paths))
        rho_bar = np.sqrt(1 - self.rho**2)
        logS = (
                -0.5 * U
                + self.rho * Z
                + rho_bar * np.cumsum(np.sqrt(np.diff(U, axis=0, prepend=U[0:1])) * normal_ort, axis=0)
        )
        return S0 * np.exp(logS), U, Z, V
