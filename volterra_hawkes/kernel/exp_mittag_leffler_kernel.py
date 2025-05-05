import numpy as np
from dataclasses import dataclass
from scipy.special import gamma

from .kernel import Kernel
from ..utility.mittag_leffler import mittag_leffler


@dataclass
class ExpMittagLefflerKernel(Kernel):
    c: float
    lam: float
    alpha: float

    def __call__(self, t):
        valid_mask = t > 0  # Avoid issues with negative values
        result = np.zeros_like(t, dtype=np.float64)
        result[valid_mask] = self.c * np.exp(-self.lam * t[valid_mask]) * t[valid_mask] ** (self.alpha - 1) * \
                             mittag_leffler(t=self.c * t[valid_mask] ** self.alpha, alpha=self.alpha, beta=self.alpha)
        return result

    def integrated_kernel(self, t):
        raise NotImplementedError()

    def double_integrated_kernel(self, t):
        raise NotImplementedError()

    @property
    def resolvent(self) -> Kernel:
        raise NotImplementedError()

    def inv_kernel(self, x):
        raise NotImplementedError

    def inv_integrated_kernel(self, x):
        raise NotImplementedError

    def inv_double_integrated_kernel(self, x):
        raise NotImplementedError
