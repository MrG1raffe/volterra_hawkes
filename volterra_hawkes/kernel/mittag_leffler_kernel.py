import numpy as np
from dataclasses import dataclass
from scipy.special import gamma

from .kernel import Kernel
from ..utility.mittag_leffler import mittag_leffler


@dataclass
class MittagLefflerKernel(Kernel):
    c: float
    alpha: float

    def __call__(self, t):
        alpha = self.alpha  # self.H + 0.5
        valid_mask = t > 0  # Avoid issues with negative values
        result = np.zeros_like(t, dtype=np.float64)
        result[valid_mask] = self.c * t[valid_mask]**(alpha - 1) * mittag_leffler(t=self.c * t[valid_mask]**alpha, alpha=alpha, beta=alpha)
        return result

    def integrated_kernel(self, t):
        valid_mask = t > 0  # Avoid issues with negative values
        result = np.zeros_like(t, dtype=np.float64)
        result[valid_mask] = self.c * t[valid_mask]**(self.alpha) * mittag_leffler(t=self.c * t[valid_mask]**self.alpha, alpha=self.alpha, beta=self.alpha+1)
        return result

    def double_integrated_kernel(self, t):
        valid_mask = t > 0  # Avoid issues with negative values
        result = np.zeros_like(t, dtype=np.float64)
        result[valid_mask] = self.c * t[valid_mask]**(self.alpha + 1) * mittag_leffler(t=self.c * t[valid_mask]**self.alpha, alpha=self.alpha, beta=self.alpha+2)
        return result

    @property
    def resolvent(self) -> Kernel:
        raise NotImplementedError

    def inv_kernel(self, x):
        raise NotImplementedError

    def inv_integrated_kernel(self, x):
        raise NotImplementedError

    def inv_double_integrated_kernel(self, x):
        raise NotImplementedError
