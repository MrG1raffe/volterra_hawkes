import numpy as np
from scipy.special import gamma, gammainc

def gamma_capped(alpha):
    """
    Compute the gamma function with a safeguard for non-positive inputs.

    Parameters
    ----------
    alpha : float
        Input value.

    Returns
    -------
    float
        Gamma(alpha) if alpha > 0, otherwise 1.
    """
    if alpha > 0:
        return gamma(alpha)
    else:
        return 1

def fractional_kernel(t, alpha, c=1):
    """
    Evaluate the fractional power kernel K(t) = c * t**(alpha - 1) / Gamma(alpha).

    Parameters
    ----------
    t : array_like
        Time points.
    alpha : float
        Exponent of the kernel.
    c : float, optional
        Scaling factor. Default is 1.

    Returns
    -------
    np.ndarray
        Kernel evaluated at each time point.
    """
    t = np.array(t)
    valid_mask = t > 0  # Avoid issues with negative values
    result = np.zeros_like(t, dtype=np.float64)
    result[valid_mask] = c * t[valid_mask]**(alpha - 1) / gamma_capped(alpha)
    return result


def inv_fractional_kernel(x, alpha, c=1):
    """
    Inverse of the fractional kernel: solves K(t) = x for t.

    Parameters
    ----------
    x : array_like
        Kernel values.
    alpha : float
        Exponent of the kernel.
    c : float, optional
        Scaling factor. Default is 1.

    Returns
    -------
    np.ndarray
        Corresponding time points t.
    """
    return (x / c * gamma_capped(alpha)) ** (1 / (alpha - 1))


def integrated_gamma_kernel(t, alpha, lam, c=1):
    """
    Compute the integral of the gamma kernel: ∫0^t c * x^(alpha-1) * exp(-λ x) dx.

    Parameters
    ----------
    t : array_like
        Time points.
    alpha : float
        Shape parameter of the gamma kernel.
    lam : float
        Rate parameter of the gamma kernel.
    c : float, optional
        Scaling factor. Default is 1.

    Returns
    -------
    np.ndarray
        Integrated gamma kernel at the time points.
    """
    return c / (lam**alpha) * gammainc(alpha, lam * t)


def double_integrated_gamma_kernel(t, alpha, lam, c = 1):
    """
    Compute the double integral of the gamma kernel: ∫0^t ∫0^s c * x^(alpha-1) * exp(-λ x) dx ds.

    Parameters
    ----------
    t : array_like
        Time points.
    alpha : float
        Shape parameter of the gamma kernel.
    lam : float
        Rate parameter of the gamma kernel.
    c : float, optional
        Scaling factor. Default is 1.

    Returns
    -------
    np.ndarray
        Double-integrated gamma kernel at the time points.
    """
    return c / (lam ** (alpha + 1)) * (lam * t * gammainc(alpha, lam * t) - gammainc(alpha + 1, lam * t) * alpha)


def mittag_leffler(t, alpha, beta, N=50):
    """
    Evaluate the Mittag-Leffler function E_{alpha, beta}(t) using series expansion.

    Parameters
    ----------
    t : array_like
        Time points.
    alpha : float
        Exponent parameter.
    beta : float
        Second parameter of the Mittag-Leffler function.
    N : int, optional
        Number of terms in the series expansion. Default is 50.

    Returns
    -------
    np.ndarray
        Mittag-Leffler function evaluated at each t.
    """
    t = np.array(t)
    ii = np.arange(N).reshape((1,) * len(t.shape) + (-1,))
    return np.sum(t.reshape(t.shape + (1,))**ii / gamma(ii * alpha + beta), axis=-1)


def integrated_exp_mittag_leffler_kernel(t, alpha, lam, c=1, N=50):
    """
    Compute the integral of the exponentially-modulated Mittag-Leffler kernel.

    ∫0^t c * x^(alpha-1) * exp(-λ x) E_{alpha, alpha}(c x^alpha) dx

    Parameters
    ----------
    t : array_like
        Time points.
    alpha : float
        Exponent parameter of the kernel.
    lam : float
        Exponential decay rate.
    c : float, optional
        Scaling factor. Default is 1.
    N : int, optional
        Number of terms in the Mittag-Leffler series. Default is 50.

    Returns
    -------
    np.ndarray
        Integrated exponential Mittag-Leffler kernel at the time points.
    """
    t = np.array(t)
    ii = np.arange(N).reshape((1,) * len(t.shape) + (-1,))
    # Numerically more stable that integrated_gamma_kernel
    return np.sum((c / (lam ** alpha))**(ii + 1) * gammainc(alpha * (ii + 1), lam * t.reshape(t.shape + (1,))), axis=-1)


def double_integrated_exp_mittag_leffler_kernel(t, alpha, lam, c=1, N=50):
    """
    Compute the double integral of the exponentially-modulated Mittag-Leffler kernel.

    Parameters
    ----------
    t : array_like
        Time points.
    alpha : float
        Exponent parameter of the kernel.
    lam : float
        Exponential decay rate.
    c : float, optional
        Scaling factor. Default is 1.
    N : int, optional
        Number of terms in the Mittag-Leffler series. Default is 50.

    Returns
    -------
    np.ndarray
        Double-integrated exponential Mittag-Leffler kernel at the time points.
    """
    t = np.array(t)
    ii = np.arange(N).reshape((1,) * len(t.shape) + (-1,))

    t_col = t.reshape(t.shape + (1,))
    alpha_arr = alpha * (ii + 1)

    return np.sum((c / (lam ** alpha))**(ii + 1) / lam * (lam * t_col * gammainc(alpha_arr, lam * t_col)
                  - gammainc(alpha_arr + 1, lam * t_col) * alpha_arr), axis=-1)