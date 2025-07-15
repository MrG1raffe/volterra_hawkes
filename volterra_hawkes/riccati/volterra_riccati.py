import numpy as np
from typing import Callable

from ..kernel.kernel import Kernel


def right_point_adams_scheme(T: float, n_steps: int, K: Kernel, F: Callable):
    t_grid = np.linspace(0, T, n_steps + 1)
    psi = np.zeros_like(t_grid)
    F_arr = np.zeros_like(t_grid)
    F_arr[0] = F(t_grid[0], psi[0])

    delta_K_bar = np.diff(K.integrated_kernel(t_grid))
    for i in range(n_steps):
        F_arr[i + 1] = F_arr[i]
        psi_pred = F_arr[1:i+2] @ np.flip(delta_K_bar[:i+1])
        F_arr[i + 1] = F(t_grid[i + 1], psi_pred)
        psi[i + 1] = F_arr[1:i+2] @ np.flip(delta_K_bar[:i+1])  # psi_pred + (F_arr[i + 1] - F_arr[i]) * delta_K_bar[i]
        F_arr[i + 1] = F(t_grid[i + 1], psi[i + 1])
    return psi, F_arr
