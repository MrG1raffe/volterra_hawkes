import numpy as np
from scipy.special import gamma, gammainc


def fractional_kernel(t, alpha, c=1):
    t = np.array(t)
    valid_mask = t > 0  # Avoid issues with negative values
    result = np.zeros_like(t, dtype=np.float64)
    result[valid_mask] = c * t[valid_mask ]**(alpha - 1) / gamma(alpha)
    return result


def inv_fractional_kernel(x, alpha, c=1):
    return (x / c * gamma(alpha)) ** (1 / (alpha - 1))


def integrated_gamma_kernel(t, alpha, lam, c=1):
    return c / (lam**alpha) * gammainc(alpha, lam * t)


def double_integrated_gamma_kernel(t, alpha, lam, c = 1):
    return c / (lam ** (alpha + 1)) * (lam * t * gammainc(alpha, lam * t) - gammainc(alpha + 1, lam * t) * alpha)

def mittag_leffler(t, alpha, beta, N=50):
    t = np.array(t)
    ii = np.arange(N).reshape((1,) * len(t.shape) + (-1,))
    return np.sum(t.reshape(t.shape + (1,))**ii / gamma(ii * alpha + beta), axis=-1)


def integrated_exp_mittag_leffler_kernel(t, alpha, lam, c=1, N=50):
    t = np.array(t)
    ii = np.arange(N).reshape((1,) * len(t.shape) + (-1,))
    # Numerically more stable that integrated_gamma_kernel
    return np.sum((c / (lam ** alpha))**(ii + 1) * gammainc(alpha * (ii + 1), lam * t.reshape(t.shape + (1,))), axis=-1)


def double_integrated_exp_mittag_leffler_kernel(t, alpha, lam, c=1, N=50):
    t = np.array(t)
    ii = np.arange(N).reshape((1,) * len(t.shape) + (-1,))

    t_col = t.reshape(t.shape + (1,))
    alpha_arr = alpha * (ii + 1)

    return np.sum((c / (lam ** alpha))**(ii + 1) / lam * (lam * t_col * gammainc(alpha_arr, lam * t_col)
                  - gammainc(alpha_arr + 1, lam * t_col) * alpha_arr), axis=-1)
    # return np.sum(double_integrated_gamma_kernel(t=t.reshape(t.shape + (1,)), alpha=alpha * (ii + 1),
    #                                              lam=lam, c=c**(ii + 1)), axis=-1)