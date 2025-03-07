import numpy as np


def simulate_hawkes(K_bar, g0_bar, t_grid, n_paths, rng=None):
    if rng is None:
        rng = np.random.default_rng(seed=42)

    n_steps = len(t_grid) - 1
    T = t_grid[-1]
    dt = T / n_steps

    # Pre-compute certain quantities independent of i and m
    K_int_dt = K_bar(dt)

    # Compute the matrix \bar K_ij
    K_bar_matrix = K_bar(t_grid[1:].reshape(-1, 1) - t_grid[:-1].reshape(1, -1)) - \
                   K_bar(t_grid[:-1].reshape(-1, 1) - t_grid[:-1].reshape(1, -1))
    K_bar_matrix = np.tril(K_bar_matrix, k=-1)

    U = np.zeros(n_paths)

    # Need to stock the Z, U now because of non-markovianity
    Z, dU = np.zeros((n_steps, n_paths)), np.zeros((n_steps, n_paths))
    g0_bar_diff = np.diff(g0_bar(t_grid))

    for i in range(n_steps):
        alpha_i = g0_bar_diff[i] + K_bar_matrix[i, :] @ Z + K_bar_matrix[i, :] @ dU
        mu = alpha_i / (1 - K_int_dt)
        lambda_ = (alpha_i / K_int_dt) ** 2

        dU_i = rng.wald(mean=mu, scale=lambda_, size=1) # inverse_gaussian_sample_vectorized(mu, lambda_, x_norm[i], x_uniform[i])
        Z_i = rng.poisson(lam=dU_i, size=n_paths) - dU_i

        Z[i, :] = Z_i
        dU[i, :] = dU_i

    dN = Z + dU
    N = np.vstack([np.zeros((1, n_paths)), np.cumsum(dN, axis=0)])
    U = np.vstack([np.zeros((1, n_paths)), np.cumsum(dU, axis=0)])

    return (dN, N, dU, U)


def simulate_volterra_rough_vectorized(H, a, b, c, rho, V_0, T, n_steps, n_paths, S_0=1.):
    # discretize time
    dt = T / n_steps

    # pre-compute certain quantities indepenendt of i and m
    K_int_dt = (dt) ** (H + 0.5) / ((H + 0.5) * gamma(H + 0.5))
    a_H = a * (dt) ** (H + 1.5) / ((H + 0.5) * (H + 1.5) * gamma(H + 0.5))
    sigma = c * K_int_dt
    rho_bar = np.sqrt(1 - rho * rho)

    ## Compute the K_matrix K_ij
    # Create arrays for broadcasting
    i_indices = np.arange(n_steps).reshape(-1, 1)
    j_indices = np.arange(n_steps).reshape(1, -1)

    K_matrix = np.maximum(1 + i_indices - j_indices, 0) ** (H + 0.5) - np.maximum(i_indices - j_indices, 0) ** (H + 0.5)
    K_matrix = np.tril(K_matrix, k=-1)  # Retain only the lower triangular part
    K_matrix *= dt ** (H + 0.5) / ((H + 0.5) * gamma(H + 0.5))

    # Initialize logS
    logS = np.zeros(n_paths)
    # V = V_0*np.ones(n_paths)
    U = np.zeros(n_paths)

    # Need to stock the Z, U now because of non-markovianity
    Z, dU = np.zeros((n_steps, n_paths)), np.zeros((n_steps, n_paths))

    for i in range(n_steps):
        tilde_alpha_i = V_0 * dt + a_H * ((i + 1) ** (H + 1.5) - i ** (H + 1.5))
        alpha_i = tilde_alpha_i + c * K_matrix[i, :] @ Z + b * K_matrix[i, :] @ dU
        mu = alpha_i / (1 - b * K_int_dt)
        lambda_ = (alpha_i / sigma) ** 2

        dU_i = inverse_gaussian_sample_vectorized(mu, lambda_, x_norm[i, :n_paths], x_uniform[i, :n_paths])
        Z_i = (1. / sigma) * ((1 - b * K_int_dt) * dU_i - alpha_i)

        logS = logS - 0.5 * dU_i + rho * Z_i + rho_bar * np.sqrt(dU_i) * x_norm2[i, :n_paths]
        U = U + dU_i

        Z[i, :] = Z_i.T
        dU[i, :] = dU_i.T

    return (S_0 * np.exp(logS), U, Z, dU)