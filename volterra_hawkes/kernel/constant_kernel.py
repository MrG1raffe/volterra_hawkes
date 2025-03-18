import numpy as np
from dataclasses import dataclass

from .kernel import Kernel


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

    def resolvent(self, t):
        return self.c * np.exp(self.c * t)
