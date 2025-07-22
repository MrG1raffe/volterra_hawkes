import numpy as np
from dataclasses import dataclass
from scipy.special import gamma

from .kernel import Kernel
from ..utility.kernel_functions import (mittag_leffler, integrated_exp_mittag_leffler_kernel,
                                        double_integrated_exp_mittag_leffler_kernel)


@dataclass
class ExpMittagLefflerKernel(Kernel):
    c: float
    lam: float
    alpha: float
    N_mittag_leffler: int = 25

    def __call__(self, t):
        valid_mask = t > 0  # Avoid issues with negative values
        result = np.zeros_like(t, dtype=np.float64)
        result[valid_mask] = self.c * np.exp(-self.lam * t[valid_mask]) * t[valid_mask] ** (self.alpha - 1) * \
                             mittag_leffler(t=self.c * t[valid_mask] ** self.alpha, alpha=self.alpha, beta=self.alpha,
                                            N=self.N_mittag_leffler)
        return result

    def integrated_kernel(self, t):
        return integrated_exp_mittag_leffler_kernel(t=t, alpha=self.alpha, lam=self.lam, c=self.c, N=self.N_mittag_leffler)

    def double_integrated_kernel(self, t):
        return double_integrated_exp_mittag_leffler_kernel(t=t, alpha=self.alpha, lam=self.lam, c=self.c, N=self.N_mittag_leffler)

    @property
    def resolvent(self) -> Kernel:
        raise NotImplementedError()

    def inv_kernel(self, x):
        raise NotImplementedError

    def inv_integrated_kernel(self, x):
        raise NotImplementedError

    def inv_double_integrated_kernel(self, x):
        raise NotImplementedError
