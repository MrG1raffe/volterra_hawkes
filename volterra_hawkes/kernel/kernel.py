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

    @abstractmethod
    def resolvent(self, t):
        pass

    #@abstractmethod
    def integrated_resolvent(self, t):
        pass

    #@abstractmethod
    def double_integrated_resolvent(self, t):
        pass