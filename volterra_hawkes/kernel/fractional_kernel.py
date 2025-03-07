import numpy as np
from dataclasses import dataclass
from scipy.special import gamma

from .kernel import Kernel
from ..utility.mittag_leffler import mittag_leffler


@dataclass
class FractionalKernel(Kernel):
    """
    Fractional kernel K(t) = c * t**(H - 0.5) / Γ(H + 0.5).
    """
    c: float = 1
    H: float = 0.1

    def __fractional_kernel(self, t, alpha: float):
        return self.c * t**(alpha - 1) / gamma(alpha)

    def kernel(self, t):
        return self.__fractional_kernel(t, alpha=self.H + 0.5)

    def integrated_kernel(self, t):
        return self.__fractional_kernel(t, alpha=self.H + 1.5)

    def double_integrated_kernel(self, t):
        return self.__fractional_kernel(t, alpha=self.H + 2.5)

    def resolvent(self, t):
        alpha = self.H + 0.5
        return self.c * t**(alpha - 1) * mittag_leffler(t=self.c * t**alpha, alpha=alpha, beta=alpha)
