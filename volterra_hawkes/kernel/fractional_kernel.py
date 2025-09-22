import numpy as np
from dataclasses import dataclass
from scipy.special import gamma

from .kernel import Kernel
from .mittag_leffler_kernel import MittagLefflerKernel
from ..utility.kernel_functions import inv_fractional_kernel, fractional_kernel


@dataclass
class FractionalKernel(Kernel):
    """
    Fractional kernel function for Volterra processes.

    The kernel exhibits power-law memory:

        K(t) = c * t^(H - 0.5) / Γ(H + 0.5)

    where `c` is a scaling constant and `H` is the Hurst parameter
    controlling the roughness or persistence of the process.

    Attributes
    ----------
    c : float, default=1
        Scaling constant of the kernel.
    H : float, default=0.1
        Hurst parameter controlling the roughness/memory of the kernel.
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
