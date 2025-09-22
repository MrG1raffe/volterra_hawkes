import numpy as np
from dataclasses import dataclass
from scipy.special import gamma

from .kernel import Kernel
from ..utility.kernel_functions import mittag_leffler


@dataclass
class MittagLefflerKernel(Kernel):
    """
    Mittag-Leffler kernel function for Volterra processes.

    The kernel combines a power-law with the generalized Mittag-Leffler function:

        K(t) = c * t^(α - 1) * E_{α, α}(c * t^α)

    where `E_{α, β}` is the generalized Mittag-Leffler function, `c` is a
    scaling constant, and `α` controls the memory and persistence of the kernel.

    Attributes
    ----------
    c : float
        Scaling constant of the kernel.
    alpha : float
        Memory parameter α controlling the power-law behavior and persistence.
    """
    c: float
    alpha: float

    def __call__(self, t):
        alpha = self.alpha  # self.H + 0.5
        valid_mask = t > 0  # Avoid issues with negative values
        result = np.zeros_like(t, dtype=np.float64)
        result[valid_mask] = self.c * t[valid_mask]**(alpha - 1) * mittag_leffler(t=self.c * t[valid_mask]**alpha, alpha=alpha, beta=alpha)
        return result

    def integrated_kernel(self, t):
        valid_mask = t > 0  # Avoid issues with negative values
        result = np.zeros_like(t, dtype=np.float64)
        result[valid_mask] = self.c * t[valid_mask]**(self.alpha) * mittag_leffler(t=self.c * t[valid_mask]**self.alpha, alpha=self.alpha, beta=self.alpha+1)
        return result

    def double_integrated_kernel(self, t):
        valid_mask = t > 0  # Avoid issues with negative values
        result = np.zeros_like(t, dtype=np.float64)
        result[valid_mask] = self.c * t[valid_mask]**(self.alpha + 1) * mittag_leffler(t=self.c * t[valid_mask]**self.alpha, alpha=self.alpha, beta=self.alpha+2)
        return result

    @property
    def resolvent(self) -> Kernel:
        raise NotImplementedError

    def inv_kernel(self, x):
        raise NotImplementedError

    def inv_integrated_kernel(self, x):
        raise NotImplementedError

    def inv_double_integrated_kernel(self, x):
        raise NotImplementedError
