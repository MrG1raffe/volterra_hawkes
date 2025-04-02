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

    def __call__(self, t):
        return self.c * np.exp(-self.lam * t)

    def integrated_kernel(self, t):
        return (self.c / self.lam) * (1 - np.exp(-self.lam * t))

    def double_integrated_kernel(self, t):
        return (self.c / self.lam) * (t - (1 - np.exp(-self.lam * t)) / self.lam)

    def resolvent(self, t):
        k = self.lam - self.c
        return self.c * np.exp(-k * t)

    def integrated_resolvent(self, t):
        k = self.lam - self.c
        if np.isclose(k, 0):
            return self.c * t
        else:
            return (self.c / k) * (1 - np.exp(-k * t))

    def double_integrated_resolvent(self, t):
        k = self.lam - self.c
        if np.isclose(k, 0):
            return 0.5 * self.c * t**2
        else:
            return (self.c / k) * (t - (1 - np.exp(-k * t)) / k)

    def inv_kernel(self, x):
        return -np.log(x / self.c) / self.lam

    def inv_integrated_kernel(self, x):
        return -np.log(1 - self.lam * x / self.c) / self.lam
