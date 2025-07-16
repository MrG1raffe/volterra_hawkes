import numpy as np
from dataclasses import dataclass
from scipy.special import gamma

from .kernel import Kernel
from .mittag_leffler_kernel import MittagLefflerKernel
from ..utility.kernel_functions import inv_fractional_kernel, fractional_kernel


@dataclass
class FractionalKernel(Kernel):
    """
    Fractional kernel K(t) = c * t**(H - 0.5) / Γ(H + 0.5).
    """
    c: float = 1
    H: float = 0.1

    def __call__(self, t):
        return fractional_kernel(t, alpha=self.H + 0.5, c=self.c)

    def integrated_kernel(self, t):
        return fractional_kernel(t, alpha=self.H + 1.5, c=self.c)

    def double_integrated_kernel(self, t):
        return fractional_kernel(t, alpha=self.H + 2.5, c=self.c)

    def inv_kernel(self, x):
        return inv_fractional_kernel(x, alpha=self.H + 0.5, c=self.c)

    def inv_integrated_kernel(self, x):
        return inv_fractional_kernel(x, alpha=self.H + 1.5, c=self.c)

    def inv_double_integrated_kernel(self, x):
        return inv_fractional_kernel(x, alpha=self.H + 2.5, c=self.c)

    @property
    def resolvent(self) -> Kernel:
        return MittagLefflerKernel(c=self.c, alpha=self.H + 0.5)
