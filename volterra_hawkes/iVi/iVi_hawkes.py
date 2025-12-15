import numpy as np
from dataclasses import dataclass
from typing import Callable

from .iVi import IVIVolterra
from ..kernel.kernel import Kernel
from ..riccati.volterra_riccati import right_point_adams_scheme


@dataclass
class IVIHawkesProcess:
    """
      Simulator for an (iVi) Hawkes process using a Volterra representation.

      This class represents a Hawkes process where the intensity
      is driven by a Volterra process. It provides methods to simulate the process
      on a discrete grid, extract arrival times, compute the mean intensity,
      and evaluate the characteristic function.

      Integrated variance dynamics
      ---------------------------
      The integrated intensity process :math:`U_t` follows a Volterra-type
      stochastic integral equation:

      .. math::

          U_t = \int_0^t g_0(s)\,ds + \int_0^t K(t-s) (\,U_s + \,Z_s )\,ds,

      where :math:`K` is the kernel function, :math:`g_0(t)` is the input function,
      and :math:`Z_t = N_t - U_t` is a Martingale auxiliary process. This defines the cumulative
      contribution of past events and the memory effect in the Hawkes process.


      Attributes
      ----------
      kernel : Kernel
          Kernel function K(t), which must implement `__call__` and `resolvent` (if `resolvent_flag` is True).
      g0_bar : Callable
          Base integrated intensity function ḡ₀(t).
      rng : numpy.random.Generator
          Random number generator used for stochastic simulations.
      resolvent_flag : bool, default=False
          Whether to use the resolvent kernel version of the iVi scheme.
      g0 : Callable, optional
          Optional function g₀(t) for instantaneous intensity.
      g0_bar_res : Callable, optional
          Optional integrated function ḡ₀(t) used when `resolvent_flag` is True.
      """
    kernel: Kernel
    g0_bar: Callable
    rng: np.random.Generator
    resolvent_flag: bool = False
    g0: Callable = None
    g0_bar_res: Callable = None

    def simulate_on_grid(self, t_grid, n_paths):
        """
        Simulate the Hawkes process on a discrete time grid.

        Uses an (iVi) Volterra process representation to generate the auxiliary
        processes (U, Z). The counting process N is then obtained as N = U + Z.

        Parameters
        ----------
        t_grid : array_like
            Discrete time points for simulation.
        n_paths : int
            Number of Monte Carlo paths.

        Returns
        -------
        N : ndarray
            Simulated counting process values of shape (len(t_grid), n_paths).
        U : ndarray
            Integrated variance/intensity process of shape (len(t_grid), n_paths).
        lam : ndarray
            Instantaneous intensity process of shape (len(t_grid), n_paths).
        """
        ivi = IVIVolterra(is_continuous=False, resolvent_flag=self.resolvent_flag, kernel=self.kernel,
                          g0_bar=self.g0_bar, g0_bar_res=self.g0_bar_res, rng=self.rng, b=1, c=1, g0=self.g0)
        U, Z, lam = ivi.simulate_u_z_v(t_grid=t_grid, n_paths=n_paths)
        N = Z + U
        return N, U, lam

    def simulate_arrivals(self, t_grid, n_paths):
        """
        Generate jump arrival times for each path of the Hawkes process.

        Parameters
        ----------
        t_grid : np.ndarray
            Discrete time points for simulation.
        n_paths : int
            Number of Monte Carlo paths.

        Returns
        -------
        arrivals : list of np.ndarray
            List of arrays containing the sorted jump times for each path.
        """
        N, U, lam = self.simulate_on_grid(t_grid=t_grid, n_paths=n_paths)
        dN = np.round(np.diff(N, axis=0)).astype(int)

        arrivals = []
        for i in range(n_paths):
            uniforms = self.rng.random(size=np.round(N[-1, i]).astype(int))
            jumps = np.repeat(t_grid[:-1], repeats=dN[:, i]) + uniforms * np.repeat(np.diff(t_grid), repeats=dN[:, i])
            arrivals.append(np.sort(jumps))

        return arrivals

    def get_mean(self, t_grid):
        """
        Compute the mean of the Hawkes process on the grid using the resolvent.

        Parameters
        ----------
        t_grid : np.ndarray
            Discrete time points.

        Returns
        -------
        mean : np.ndarray
            Mean values of the process at each time point.
        """
        R_mat = np.tril(self.kernel.resolvent(t_grid[:, None] - t_grid[None, :]), k=-1)
        mean = self.g0_bar(t_grid) + R_mat @ self.g0_bar(t_grid) * (t_grid[1] - t_grid[0])
        return mean

    def characteristic_function(self, T, w, n_steps, mode: str = "U"):
        """
        Compute the characteristic function of the process at time T.

        Parameters
        ----------
        T : float
            Terminal time.
        w : complex
            Argument of the characteristic function.
        n_steps : int
            Number of discretization steps for the integration.
        mode : {'U', 'N'}, default='U'
            Compute the characteristic function for the integrated process U or
            the counting process N.

        Returns
        -------
        cf : complex
            Value of the characteristic function at w.
        """
        if mode == "U":
            F = lambda t, p: w + np.exp(p) - 1
        elif mode == "N":
            F = lambda t, p: np.exp(w + p) - 1
        else:
            raise ValueError("Mode can take only the values `U` and `N`.")
        psi, F_arr = right_point_adams_scheme(T=T, n_steps=n_steps, K=self.kernel, F=F)
        t_grid = np.linspace(0, T, n_steps + 1)
        dG_0 = np.diff(self.g0_bar(t_grid))
        return np.exp(np.flip(F_arr[1:]) @ dG_0)
