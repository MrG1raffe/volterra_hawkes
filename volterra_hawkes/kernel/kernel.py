from __future__ import annotations
from abc import ABC, abstractmethod

class Kernel(ABC):
    """
    Abstract base class for a kernel function used in Volterra-type processes.

    A kernel represents a memory function K(t) that defines interactions
    over time in Volterra integral equations. Concrete subclasses must
    implement the kernel function, its integrals, and optionally its inverse.
    """

    @abstractmethod
    def __call__(self, t):
        """
        Evaluate the kernel function K at a given time.

        Parameters
        ----------
        t : float or np.ndarray
            Time or array of time points.

        Returns
        -------
        float or np.ndarray
            Kernel value(s) at the given time(s).
        """
        pass

    @abstractmethod
    def integrated_kernel(self, t):
        """
        Compute the integral of the kernel from 0 to t.

        Parameters
        ----------
        t : float or np.ndarray
            Upper limit(s) of integration.

        Returns
        -------
        float or np.ndarray
            Integral of the kernel over [0, t].
        """
        pass

    @abstractmethod
    def double_integrated_kernel(self, t):
        """
        Compute the double integral of the kernel from 0 to t.

        Parameters
        ----------
        t : float or np.ndarray
            Upper limit(s) of integration.

        Returns
        -------
        float or np.ndarray
            Double integral of the kernel over [0, t].
        """
        pass

    @property
    @abstractmethod
    def resolvent(self) -> Kernel:
        """
        Return the resolvent of the second kind for the kernel.

        The resolvent R satisfies the convolution equation:

        K ★ R = R - K

        where ★ denotes convolution.

        Returns
        -------
        Kernel
            The resolvent kernel as an instance of `Kernel`.
        """
        pass

    @abstractmethod
    def inv_kernel(self, x):
        """
        Compute the functional inverse of the kernel.

        Parameters
        ----------
        x : float or np.ndarray
            Kernel value(s) to invert.

        Returns
        -------
        float or np.ndarray
            Inverse kernel value(s).
        """
        pass

    @abstractmethod
    def inv_integrated_kernel(self, x):
        """
        Compute the functional inverse of the integrated kernel.

        Parameters
        ----------
        x : float or np.ndarray
            Integrated kernel value(s) to invert.

        Returns
        -------
        float or np.ndarray
            Inverse integrated kernel value(s).
        """
        pass

    @abstractmethod
    def inv_double_integrated_kernel(self, x):
        """
        Compute the functional inverse of the double integrated kernel.

        Parameters
        ----------
        x : float or np.ndarray
            Double integrated kernel value(s) to invert.

        Returns
        -------
        float or np.ndarray
            Inverse double integrated kernel value(s).
        """
        pass
