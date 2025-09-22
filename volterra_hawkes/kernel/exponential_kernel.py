import numpy as np
from dataclasses import dataclass

from .kernel import Kernel


@dataclass
class ExponentialKernel(Kernel):
    """
    Exponential kernel function for Volterra processes.

    The kernel decays exponentially over time:

        K(t) = c * exp(-λ * t)

    where `c` is a scaling constant and `λ` controls the decay rate.

    Attributes
    ----------
    c : float, default=1
        Scaling constant of the kernel.
    lam : float, default=1
        Decay rate λ of the exponential kernel.
    """
    c: float = 1
    lam: float = 1

    def __call__(self, t):
        return self.c * np.exp(-self.lam * t)

    def integrated_kernel(self, t):
        if np.isclose(self.lam, 0):
            return self.c * t
        else:
            return (self.c / self.lam) * (1 - np.exp(-self.lam * t))

    def double_integrated_kernel(self, t):
        if np.isclose(self.lam, 0):
            return 0.5 * self.c * t**2
        else:
            return (self.c / self.lam) * (t - (1 - np.exp(-self.lam * t)) / self.lam)

    @property
    def resolvent(self) -> Kernel:
        return ExponentialKernel(c=self.c, lam=self.lam - self.c)

    def inv_kernel(self, x):
        return -np.log(x / self.c) / self.lam

    def inv_integrated_kernel(self, x):
        return -np.log(1 - self.lam * x / self.c) / self.lam

    def inv_double_integrated_kernel(self, x):
        raise NotImplementedError
