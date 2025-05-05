import numpy as np
from dataclasses import dataclass
from scipy.special import gamma, gammainc, gammaincinv

from .kernel import Kernel
from .exp_mittag_leffler_kernel import ExpMittagLefflerKernel


@dataclass
class GammaKernel(Kernel):
    """
    Gamma kernel K(t) = c * e**(-lam * x) * t**(alpha - 1) / Γ(alpha).
    """
    alpha: float
    lam: float
    c: float = 1

    def __fractional_kernel(self, t, alpha: float):
        t = np.array(t)
        valid_mask = t > 0  # Avoid issues with negative values
        result = np.zeros_like(t, dtype=np.float64)
        result[valid_mask] = self.c * t[valid_mask]**(alpha - 1) / gamma(alpha)
        return result

    def __call__(self, t):
        return self.__fractional_kernel(t, alpha=self.alpha) * np.exp(-self.lam * t)

    def integrated_kernel(self, t):
        return self.c / (self.lam**self.alpha) * gammainc(self.alpha, self.lam * t)

    def double_integrated_kernel(self, t):
        return self.c / (self.lam**(self.alpha + 1)) * (self.lam * t * gammainc(self.alpha, self.lam * t) -
                                                        gammainc(self.alpha + 1, self.lam * t) * self.alpha)

    @property
    def resolvent(self) -> Kernel:
        return ExpMittagLefflerKernel(c=self.c, lam=self.lam, alpha=self.alpha)

    def inv_kernel(self, x):
        raise NotImplementedError

    def inv_integrated_kernel(self, x):
        return gammaincinv(self.alpha, x * (self.lam**self.alpha) / self.c) / self.lam

    def inv_double_integrated_kernel(self, x):
        raise NotImplementedError
