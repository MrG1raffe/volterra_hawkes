import numpy as np
from dataclasses import dataclass
from scipy.special import gamma, gammainc, gammaincinv

from .kernel import Kernel
from .exp_mittag_leffler_kernel import ExpMittagLefflerKernel
from ..utility.kernel_functions import fractional_kernel, integrated_gamma_kernel, double_integrated_gamma_kernel


@dataclass
class GammaKernel(Kernel):
    """
    Gamma kernel K(t) = c * e**(-lam * x) * t**(alpha - 1) / Γ(alpha).
    """
    alpha: float
    lam: float
    c: float = 1

    def __call__(self, t):
        return fractional_kernel(t, alpha=self.alpha, c=self.c) * np.exp(-self.lam * t)

    def integrated_kernel(self, t):
        return integrated_gamma_kernel(t, alpha=self.alpha, lam=self.lam, c=self.c)

    def double_integrated_kernel(self, t):
        return double_integrated_gamma_kernel(t=t, alpha=self.alpha, lam=self.lam, c=self.c)

    @property
    def resolvent(self) -> Kernel:
        return ExpMittagLefflerKernel(c=self.c, lam=self.lam, alpha=self.alpha)

    def inv_kernel(self, x):
        raise NotImplementedError

    def inv_integrated_kernel(self, x):
        return gammaincinv(self.alpha, x * (self.lam**self.alpha) / self.c) / self.lam

    def inv_double_integrated_kernel(self, x):
        raise NotImplementedError
