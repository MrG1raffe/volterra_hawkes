import numpy as np
from dataclasses import dataclass
from scipy.special import gamma

from .kernel import Kernel
from .mittag_leffler_kernel import MittagLefflerKernel
from ..utility.mittag_leffler import mittag_leffler


@dataclass
class FractionalKernel(Kernel):
    """
    Fractional kernel K(t) = c * t**(H - 0.5) / Γ(H + 0.5).
    """
    c: float = 1
    H: float = 0.1

    def __fractional_kernel(self, t, alpha: float):
        t = np.array(t)
        valid_mask = t > 0  # Avoid issues with negative values
        result = np.zeros_like(t, dtype=np.float64)
        result[valid_mask] = self.c * t[valid_mask]**(alpha - 1) / gamma(alpha)
        return result

    def __inv_fractional_kernel(self, x, alpha: float):
        return (x / self.c * gamma(alpha)) ** (1 / (alpha - 1))

    def __call__(self, t):
        return self.__fractional_kernel(t, alpha=self.H + 0.5)

    def integrated_kernel(self, t):
        return self.__fractional_kernel(t, alpha=self.H + 1.5)

    def double_integrated_kernel(self, t):
        return self.__fractional_kernel(t, alpha=self.H + 2.5)

    def resolvent(self, t):
        alpha = self.H + 0.5
        valid_mask = t > 0  # Avoid issues with negative values
        result = np.zeros_like(t, dtype=np.float64)
        result[valid_mask] = self.c * t[valid_mask]**(alpha - 1) * mittag_leffler(t=self.c * t[valid_mask]**alpha, alpha=alpha, beta=alpha)
        return result

    def inv_kernel(self, x):
        return self.__inv_fractional_kernel(x, alpha=self.H + 0.5)

    def inv_integrated_kernel(self, x):
        return self.__inv_fractional_kernel(x, alpha=self.H + 1.5)

    def inv_double_integrated_kernel(self, x):
        return self.__inv_fractional_kernel(x, alpha=self.H + 2.5)
    
    def resolvent_as_kernel(self):
        return MittagLefflerKernel(c=self.c, H=self.H)
