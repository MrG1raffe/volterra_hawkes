from dataclasses import dataclass
from .kernel import Kernel

@dataclass
class ShiftedKernel(Kernel):
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