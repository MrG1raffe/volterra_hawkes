import numpy as np
from dataclasses import dataclass

from .kernel import Kernel


@dataclass
class ExponentialKernel(Kernel):
    """
    Exponential kernel K(t) = c * exp(-lam * t).
    """
    c: float = 1
    lam: float = 1

    def kernel(self, t):
        return self.c * np.exp(-self.lam * t)

    def integrated_kernel(self, t):
        return (self.c / self.lam) * (1 - np.exp(-self.lam * t))

    def double_integrated_kernel(self, t):
        return (self.c / self.lam) * (t - (1 - np.exp(-self.lam * t)) / self.lam)

    def resolvent(self, t):
        return self.c * np.exp((self.c - self.lam) * t)
