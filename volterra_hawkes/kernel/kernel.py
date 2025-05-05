from __future__ import annotations
from abc import ABC, abstractmethod


class Kernel(ABC):
    @abstractmethod
    def __call__(self, t):
        pass

    @abstractmethod
    def integrated_kernel(self, t):
        pass

    @abstractmethod
    def double_integrated_kernel(self, t):
        pass

    @property
    @abstractmethod
    def resolvent(self) -> Kernel:
        """
        A resolvent of the second kind R satisfying the resolvent equation
        K ★ R = R - K,
        where ★ stands for convolution.

        :return: The resolvent as an instance of `Kernel`.
        """
        pass

    @abstractmethod
    def inv_kernel(self, x):
        pass

    @abstractmethod
    def inv_integrated_kernel(self, x):
        pass

    @abstractmethod
    def inv_double_integrated_kernel(self, x):
        pass
