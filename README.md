# iVi Volterra and Hawkes Process Simulation Module

This Python module provides a comprehensive framework for simulating interacting Volterra (iVi) processes and Hawkes processes, including their integrated and exponential-variant forms. The package supports a wide variety of kernels, including fractional, Mittag-Leffler, exponential, gamma, and sum-of-exponentials kernels.

## Features

- **iVi Volterra simulation**:
Simulate stock prices and integrated variance processes driven by Volterra-type kernels.

- **Hawkes process simulation**:
Exact and approximate methods for simulating standard and exponential Hawkes processes.

- **Flexible kernel framework**:
  - Integrated kernel functions and inverses
  - Resolvents of the second kind

- **Utilities for numerical solutions**
  - Numerical scheme for Volterra equations

- **Poisson and inhomogeneous Poisson simulation**

## Installation
```
git clone <repo-url>
cd <repo-folder>
pip install -r requirements.txt
```

## Usage
### iVi Volterra Simulation
```
import numpy as np
from volterra_hawkes.iVi.iVi_volterra_vol_model import IVIVolterraVolModel
from volterra_hawkes.kernel.kernels import FractionalKernel

# Model parameters
H = 0.1
a, b, c = 0.02, -0.3, 0.3
V0 = 0.02
rho = -0.7
T = 1
n_steps = 300
n_paths = 10

t_grid = np.linspace(0, T, n_steps + 1)
kernel = FractionalKernel(H=H)

def g0(t):
    return V0 + a * kernel.integrated_kernel(t)

def g0_bar(t):
    return V0 * t + a * kernel.double_integrated_kernel(t)

rng = np.random.default_rng(seed=42)
model = IVIVolterraVolModel(
    is_continuous=True,
    resolvent_flag=False,
    kernel=kernel,
    g0_bar=g0_bar,
    g0=g0,
    b=b,
    c=c,
    rho=rho,
    rng=rng
)

S, U, Z, V = model.simulate_price(t_grid=t_grid, n_paths=n_paths)
```

### Hawkes Process Simulation
```
from volterra_hawkes.iVi.iVi_hawkes import IVIHawkesProcess
from volterra_hawkes.kernel.kernels import ExponentialKernel

kernel = ExponentialKernel(c=0.1, lam=0.5)
hawkes = IVIHawkesProcess(kernel=kernel, g0_bar=lambda t: 0.02, rng=np.random.default_rng(42))

t_grid = np.linspace(0, 1, 100)
N, U, lam = hawkes.simulate_on_grid(t_grid, n_paths=10)
arrivals = hawkes.simulate_arrivals(t_grid, n_paths=10)
```

### Poisson Utilities

- `poisson_field_on_interval(a, b, intensity, rng)` – simulate a homogeneous Poisson process.

- `inhomogeneous_poisson_field_on_interval_thinning(a, b, intensity, intensity_bound, rng)` – simulate an inhomogeneous Poisson process using thinning.

- `inhomogeneous_poisson_field_on_interval_inversion(a, b, integrated_intensity, inv_integrated_intensity, rng)` – simulate an inhomogeneous Poisson process using inversion.

### Kernel Utilities

- Constant, Fractional, Mittag-Leffler, Exponential, Gamma, Sum-of-Exponential kernels.

- Integrated and double-integrated versions for numerical approximation.

- Inverse kernel functions for Poisson inversion methods.
- Resolvents of the second kind.

### Volterra Solver

- `right_point_adams_scheme(T, n_steps, K, F)` – numerical solution for Volterra integral equations.

### Jump-based Computations

- `lam_from_jumps(t, t_jumps, kernel, g0)` – intensity function from Hawkes jumps.

- `U_from_jumps(t, t_jumps, kernel, g0_bar)` – integrated intensity.

- `N_from_jumps(t, t_jumps)` – counting process from jumps.


