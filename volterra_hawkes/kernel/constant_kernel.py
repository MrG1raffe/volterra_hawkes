import numpy as np
from dataclasses import dataclass

from .kernel import Kernel
from .exponential_kernel import ExponentialKernel


@dataclass
class ConstantKernel(Kernel):
    """
    Constant kernel K(t) = c.
    """
    c: float = 1

    def __call__(self, t):
        return self.c * np.ones_like(t)

    def integrated_kernel(self, t):
        return self.c * t

    def double_integrated_kernel(self, t):
        return 0.5 * self.c * t**2

    @property
    def resolvent(self) -> Kernel:
        return ExponentialKernel(c=self.c, lam=-self.c)

    def inv_integrated_kernel(self, x):
        return x / self.c

    def inv_kernel(self, x):
        raise NotImplementedError

    def inv_double_integrated_kernel(self, x):
        raise NotImplementedError
