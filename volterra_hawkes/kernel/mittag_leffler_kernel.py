import numpy as np
from dataclasses import dataclass
from scipy.special import gamma

from .kernel import Kernel
from ..utility.mittag_leffler import mittag_leffler


@dataclass
class MittagLefflerKernel(Kernel):
    c: float
    H: float

    def __call__(self, t):
        alpha = self.H + 0.5
        valid_mask = t > 0  # Avoid issues with negative values
        result = np.zeros_like(t, dtype=np.float64)
        result[valid_mask] = self.c * t[valid_mask]**(alpha - 1) * mittag_leffler(t=self.c * t[valid_mask]**alpha, alpha=alpha, beta=alpha)
        return result

    def integrated_kernel(self, t):
        alpha = self.H + 0.5
        valid_mask = t > 0  # Avoid issues with negative values
        result = np.zeros_like(t, dtype=np.float64)
        result[valid_mask] = self.c * t[valid_mask]**(alpha) * mittag_leffler(t=self.c * t[valid_mask]**alpha, alpha=alpha, beta=alpha+1)
        return result

    def double_integrated_kernel(self, t):
        alpha = self.H + 0.5
        valid_mask = t > 0  # Avoid issues with negative values
        result = np.zeros_like(t, dtype=np.float64)
        result[valid_mask] = self.c * t[valid_mask]**(alpha + 1) * mittag_leffler(t=self.c * t[valid_mask]**alpha, alpha=alpha, beta=alpha+2)
        return result

    def resolvent(self, t):
        return t

    def resolvent_as_kernel(self):
        pass