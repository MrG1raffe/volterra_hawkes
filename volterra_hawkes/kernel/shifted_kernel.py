from dataclasses import dataclass
from .kernel import Kernel

@dataclass
class ShiftedKernel(Kernel):
    """
    Shifted kernel function for Volterra processes.

    This kernel applies a time shift to an existing kernel:

        K_shifted(t) = K(t + ε)

    where `K` is the base kernel and `ε` is a positive shift parameter.
    The integrated and double integrated kernels are adjusted to account
    for the shift.

    Attributes
    ----------
    eps : float
        Horizontal shift applied to the base kernel.
    kernel : Kernel
        Base kernel instance that is being shifted.
    """
    eps: float
    kernel: Kernel

    def __call__(self, t):
        return self.kernel(t + self.eps)

    def integrated_kernel(self, t):
        return self.kernel.integrated_kernel(t + self.eps) - self.kernel.integrated_kernel(self.eps)

    def double_integrated_kernel(self, t):
        return (self.kernel.double_integrated_kernel(t + self.eps) - self.kernel.double_integrated_kernel(self.eps)
                - t * self.kernel.integrated_kernel(self.eps))

    @property
    def resolvent(self) -> Kernel:
        raise NotImplementedError

    def inv_kernel(self, x):
        return self.kernel.inv_kernel(x) - self.eps

    def inv_integrated_kernel(self, x):
        return self.kernel.inv_integrated_kernel(x + self.kernel.integrated_kernel(self.eps)) - self.eps

    def inv_double_integrated_kernel(self, x):
        raise NotImplementedError