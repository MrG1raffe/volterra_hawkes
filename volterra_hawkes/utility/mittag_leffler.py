import numpy as np
from scipy.special import gamma


def mittag_leffler(t, alpha, beta, N=50):
    t = np.array(t)
    ii = np.arange(N).reshape((1,) * len(t.shape) + (-1,))
    return np.sum(t.reshape(t.shape + (1,))**ii / gamma(ii * alpha + beta), axis=-1)