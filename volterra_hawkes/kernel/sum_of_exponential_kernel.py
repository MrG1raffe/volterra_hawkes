import numpy as np
from numpy.typing import NDArray
from numpy import float_
from numpy.polynomial.polynomial import polyfromroots
from scipy.linalg import circulant
from scipy.interpolate import lagrange
from dataclasses import dataclass
from typing import Union, Tuple

from .kernel import Kernel


@dataclass
class SumOfExponentialKernel(Kernel):
    """
    Exponential kernel K(t) = sum_i c_i * exp(-lam_i * t).
    """
    c: np.ndarray = np.ones(1)
    lam: np.ndarray = np.zeros(1)

    def __call__(self, t: NDArray[float_]):
        return (self.c * np.exp(-self.lam * t[..., np.newaxis])).sum(axis=-1)

    def integrated_kernel(self, t: NDArray[float_]):
        return ((self.c / self.lam) * (1 - np.exp(-self.lam * t[..., np.newaxis]))).sum(axis=-1)

    def double_integrated_kernel(self, t: NDArray[float_]):
        return ((self.c / self.lam) * (t[..., np.newaxis] -
                                       (1 - np.exp(-self.lam * t[..., np.newaxis])) / self.lam)).sum(axis=-1)

    @property
    def resolvent(self) -> Kernel:
        alphas, betas = self.__compute_alphas_and_betas()
        return SumOfExponentialKernel(c=alphas, lam=betas)

    def __compute_alphas_and_betas(self) -> Tuple:
        """
        The resolvent of the second kind satisfying
        R + µ K ★ R + µ K = 0  (star denotes convolution).
        Our resolvent corresponds to µ = -1.
        In the case where K is the sum of exponential kernels, one can reduce this convolution equation to the ODE
        of order n_stochastic_factors. Its solution can be found in the form R(t) = Σ α_i * exp(-β_i * t).
        The function computes arrays (α_i) and (β_i).

        :return: (α_i), (β_i) as arrays.
        """
        mu = -1
        n_factors = self.c.size
        if n_factors == 1:
            # explicit formula for L=1 is available
            betas = (mu * self.c + self.lam)
            alphas = -mu * self.c
        elif n_factors == 2:
            # explicit formula for L=1 is available
            B = np.sum(self.lam) + mu * self.__k_der_0(0)
            C = np.prod(self.lam) + np.sum(self.lam) * mu * self.__k_der_0(0) + mu * self.__k_der_0(1)
            D = B ** 2 - 4 * C

            # roots of the characteristic polynomial for R
            betas = 0.5 * (-B - np.sqrt(D)) * np.ones(2)
            betas[1] += np.sqrt(D)

            b = [-mu * self.__k_der_0(0), -mu * self.__k_der_0(1) + mu ** 2 * self.__k_der_0(0) ** 2]
            alphas = np.array([b[0] * betas[1] - b[1], b[1] - b[0] * betas[0]]) / (betas[1] - betas[0])
            betas *= -1
        else:

            # in general case, the roots of the characteristic polynomial are found numerically.
            K_der_lam = self.__k_der_0(np.arange(n_factors)) * mu
            p = polyfromroots(-self.lam)
            P = np.flip(np.triu(circulant(np.flip(p)).T), axis=1)
            # construct the coefficients of the ODE on R
            betas_polynom = P @ np.concatenate([[1], K_der_lam])
            betas = np.roots(np.flip(betas_polynom))

            # define the initial conditions of R^(l) for l = 0, ..., L.
            R_der_0 = np.zeros(len(betas) + 1)
            R_der_0[0] = 1
            L = np.zeros((n_factors, n_factors))
            for i in range(len(betas)):
                R_der_0[i + 1] = -K_der_lam[:i + 1] @ np.flip(R_der_0[:i + 1])
                e_i = np.zeros_like(betas)
                e_i[i] = 1.0
                # rows of the inverse Vandermonde matrix correspond to the coefficients of the Lagrange polynomials
                L[i] = lagrange(betas, e_i).coef[::-1]
            alphas = L @ R_der_0[1:]
            betas *= -1
        return alphas.squeeze(), betas.squeeze()

    def __k_der_0(self, der_order: Union[int, NDArray[float_]]):
        """
        Calculates the kernel derivative at t=0 of order `der_order`.

        :param der_order: derivative order.
        :return: value of K^(der_order)(0).
        """
        return np.sum(self.c * (-self.lam) ** np.reshape(der_order, (-1, 1)), axis=1)

    def inv_kernel(self, x):
        raise NotImplementedError

    def inv_integrated_kernel(self, x):
        raise NotImplementedError

    def inv_double_integrated_kernel(self, x):
        raise NotImplementedError
