from dataclasses import dataclass
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from statsmodels.graphics.gofplots import qqplot_2samples
from scipy.stats import ks_2samp, kstest
from scipy.stats import expon
import statsmodels.api as sm
from typing import Callable, Union

from .sim_counter import SimCounter
from ..kernel.kernel import Kernel
from ..point_processes.hawkes import simulate_hawkes, simulate_hawkes_ogata, U_from_jumps, lam_from_jumps, N_from_jumps
from ..iVi.iVi_hawkes import IVIHawkesProcess


DEFAULT_COLOR_CYCLE = ("#B56246", "#579F40", "#9A46B5", "#4699B5", "#B54662", "#D4A017", "#5b6c64", "#71074E", "#0033A0")


@dataclass
class Experiment:
    T: float
    n_steps: int
    kernel: Kernel
    mu: float
    decreasing_kernel: Union[Callable, Kernel] = None

    def __post_init__(self):
        self.g0 = lambda t: self.mu * np.ones_like(t)
        self.g0_bar = lambda t: self.mu * t
        self.g0_bar_res = lambda t: self.mu * t + self.mu * self.kernel.resolvent.double_integrated_kernel(t)
        self.t_grid = np.linspace(0, self.T, self.n_steps + 1)

    def change_n_steps(self, n_steps):
        res = Experiment(T=self.T, n_steps=n_steps, kernel=self.kernel, mu=self.mu, decreasing_kernel=self.decreasing_kernel)
        return res


def get_N_U_sample(experiment: Experiment, method: str, n_paths: int, rng=None, return_counters: bool = False):
    if rng is None:
        rng = np.random.default_rng(seed=42)
    if method == "Population":
        sim_counters = [SimCounter() for _ in range(n_paths)]
        hawkes_arrivals_sample = [simulate_hawkes(T=experiment.T, g0=experiment.g0, g0_upper_bound=experiment.mu,
                                                  kernel=experiment.kernel, rng=rng, sim_counter=sim_counter) for
                                  sim_counter in sim_counters]
        N_sample = np.array(
            [N_from_jumps(np.array(experiment.t_grid), hawkes_arrivals) for hawkes_arrivals in hawkes_arrivals_sample])
        U_sample = np.array([U_from_jumps(np.array(experiment.t_grid), hawkes_arrivals, kernel=experiment.kernel,
                                          g0_bar=experiment.g0_bar) for hawkes_arrivals in hawkes_arrivals_sample])
        if return_counters:
            return N_sample, U_sample, [sim_counter.counter for sim_counter in sim_counters]
        else:
            return N_sample, U_sample
    elif method == "Ogata":
        sim_counters = [SimCounter() for _ in range(n_paths)]
        hawkes_arrivals_sample = [simulate_hawkes_ogata(T=experiment.T, mu=experiment.mu, kernel=experiment.kernel,
                                                        rng=rng, batch_size=1, sim_counter=sim_counter,
                                                        decreasing_kernel=experiment.decreasing_kernel)
                                  for sim_counter in sim_counters]
        N_sample = np.array(
            [N_from_jumps(np.array(experiment.t_grid), hawkes_arrivals) for hawkes_arrivals in hawkes_arrivals_sample])
        U_sample = np.array([U_from_jumps(np.array(experiment.t_grid), hawkes_arrivals, kernel=experiment.kernel,
                                          g0_bar=experiment.g0_bar) for hawkes_arrivals in hawkes_arrivals_sample])
        if return_counters:
            return N_sample, U_sample, [sim_counter.counter for sim_counter in sim_counters]
        else:
            return N_sample, U_sample
    elif method == "iVi":
        ivi_hawkes = IVIHawkesProcess(kernel=experiment.kernel, g0_bar=experiment.g0_bar, rng=rng, g0=experiment.g0,
                                      resolvent_flag=False)
        N, U, lam = ivi_hawkes.simulate_on_grid(t_grid=experiment.t_grid, n_paths=n_paths)
        return N.T, U.T
    elif method == "Res iVi":
        ivi_hawkes_res = IVIHawkesProcess(kernel=experiment.kernel, g0_bar=experiment.g0_bar,
                                          g0_bar_res=experiment.g0_bar_res, rng=rng, g0=experiment.g0,
                                          resolvent_flag=True)
        N_res, U_res, lam_res = ivi_hawkes_res.simulate_on_grid(t_grid=experiment.t_grid, n_paths=n_paths)
        return N_res.T, U_res.T


def get_arrivals_sample(experiment: Experiment, method: str, n_paths: int, rng=None):
    if rng is None:
        rng = np.random.default_rng(seed=42)
    if method == "Population":
        hawkes_arrivals_sample = [
            simulate_hawkes(T=experiment.T, g0=experiment.g0, g0_upper_bound=experiment.mu, kernel=experiment.kernel,
                            rng=rng) for _ in range(n_paths)]
        return hawkes_arrivals_sample
    elif method == "Ogata":
        hawkes_arrivals_sample = [
            simulate_hawkes_ogata(T=experiment.T, mu=experiment.mu, kernel=experiment.kernel, rng=rng, batch_size=400,
                                  decreasing_kernel=experiment.decreasing_kernel)
            for _ in range(n_paths)]
        return hawkes_arrivals_sample
    elif method == "iVi":
        ivi_hawkes = IVIHawkesProcess(kernel=experiment.kernel, g0_bar=experiment.g0_bar, rng=rng, g0=experiment.g0,
                                      resolvent_flag=False)
        return ivi_hawkes.simulate_arrivals(t_grid=experiment.t_grid, n_paths=n_paths)
    elif method == "Res iVi":
        ivi_hawkes_res = IVIHawkesProcess(kernel=experiment.kernel, g0_bar=experiment.g0_bar,
                                          g0_bar_res=experiment.g0_bar_res, rng=rng, g0=experiment.g0,
                                          resolvent_flag=True)
        return ivi_hawkes_res.simulate_arrivals(t_grid=experiment.t_grid, n_paths=n_paths)


def plot_trajectories(e, n_paths=10, path: str = None):
    rng = np.random.default_rng(seed=42)

    ivi_hawkes = IVIHawkesProcess(kernel=e.kernel, g0_bar=e.g0_bar, rng=rng, g0=e.g0)
    ivi_hawkes_res = IVIHawkesProcess(kernel=e.kernel, g0_bar=e.g0_bar_res, g0_bar_res=e.g0_bar_res, rng=rng, g0=e.g0,
                                      resolvent_flag=True)

    N, U, lam = ivi_hawkes.simulate_on_grid(t_grid=e.t_grid, n_paths=n_paths)
    N_res, U_res, lam_res = ivi_hawkes_res.simulate_on_grid(t_grid=e.t_grid, n_paths=n_paths)

    fig, ax = plt.subplots(2, 3, figsize=(13, 6))
    n_show = n_paths

    ax[0, 0].plot(e.t_grid, N[:, 0:n_show])
    ax[0, 0].set_title('Hawkes process N')
    ax[0, 0].grid("on")

    ax[0, 1].plot(e.t_grid, U[:, 0:n_show])
    ax[0, 1].set_title('Integrated intensity U')
    ax[0, 1].grid("on")

    ax[0, 2].plot(e.t_grid, lam[:, 0:n_show])
    ax[0, 2].set_title('Instantaneous intensity $\lambda$')
    ax[0, 2].grid("on")

    ax[1, 0].plot(e.t_grid, N_res[:, 0:n_show])
    ax[1, 0].set_title('Hawkes process N (res)')
    ax[1, 0].grid("on")

    ax[1, 1].plot(e.t_grid, U_res[:, 0:n_show])
    ax[1, 1].set_title('Integrated intensity U (res)')
    ax[1, 1].grid("on")

    ax[1, 2].plot(e.t_grid, lam_res[:, 0:n_show])
    ax[1, 2].set_title('Instantaneous intensity $\lambda$ (res)')
    ax[1, 2].grid("on")

    if path is not None:
        fig.savefig(fname=path, format="pdf", bbox_inches="tight", transparent=True)


def plot_marginal_laws(samples, flag, methods=("Population", "Ogata", "iVi", "Res iVi"), path=None):
    if flag == "N":
        idx = 0
    elif flag == "U":
        idx = 1
    else:
        raise ValueError("flag must be 'N' or 'U'")

    fig, ax = plt.subplots(3, 2, figsize=(13, 9))

    X_T_exact = samples["Population"][idx][:, -1]
    # x_grid = np.linspace(2, 4, 1000)
    x_grid = np.linspace(0, np.max(X_T_exact), 1000)
    ecdf_exact = sm.distributions.ECDF(samples["Population"][idx][:, -1])
    ax[0, 0].plot(x_grid, ecdf_exact(x_grid), label=methods[0], color="k")

    p_values_dict = dict()
    for method, ax_qq in zip(methods[1:], [ax[1, 1], ax[2, 0], ax[2, 1]]):
        X_T = samples[method][idx][:, -1]
        ecdf = sm.distributions.ECDF(X_T)
        # x_grid = np.linspace(2, 4, 1000)
        x_grid = np.linspace(0, max(np.max(X_T), np.max(X_T_exact)), 1000)
        ax[0, 0].plot(x_grid, ecdf(x_grid), label=method)
        ax[1, 0].hist(X_T, density=True, bins=75, alpha=0.3, label=method)
        ax[0, 1].plot(x_grid, np.abs(ecdf(x_grid) - ecdf_exact(x_grid)), label=method)
        # print(method, np.max(np.abs(ecdf(x_grid) - ecdf_exact(x_grid))))
        qqplot_2samples(sm.ProbPlot(X_T_exact), sm.ProbPlot(X_T), ax=ax_qq, xlabel="Population", ylabel=method,
                        line="45")

        print(f"p-value Population-{method}:", ks_2samp(X_T, X_T_exact).pvalue)
        # print(ks_2samp(X_T, X_T_exact))
        p_values_dict[f"Population-{method}"] = ks_2samp(X_T, X_T_exact).pvalue, ks_2samp(X_T, X_T_exact).statistic

    ax[0, 0].legend()
    ax[0, 1].legend()
    ax[1, 0].legend()

    if path is not None:
        fig.savefig(fname=path, format="pdf", bbox_inches="tight", transparent=True)

    return p_values_dict


def poisson_jumps_test(jumps, path=None,
                       color_cycle=DEFAULT_COLOR_CYCLE,
                       ax: matplotlib.axes = None):
    data = np.diff(jumps, prepend=0)
    data_unif = 1 - np.exp(-data)

    # Compute theoretical quantiles (from an Exponential(1) distribution)
    n = len(data)
    empirical_quantiles = np.sort(data)
    theoretical_quantiles = expon.ppf((np.arange(1, n + 1) - 0.5) / n,
                                      scale=1.0)  # Inverse CDF (percent-point function)

    # Q-Q plot
    if ax is None:
        fig, ax = plt.subplots(1, 3, figsize=(10, 3))
    ax[0].scatter(theoretical_quantiles, empirical_quantiles, label="Observed vs. Exponential", s=10)
    ax[0].plot(theoretical_quantiles, theoretical_quantiles, c=color_cycle[1], linestyle="dashed", label="y = x")

    ax[0].set_xlabel("Theoretical Quantiles (Exponential)")
    ax[0].set_ylabel("Empirical Quantiles (Data)")
    ax[0].set_title("Q-Q Plot Against Exponential Distribution")
    ax[0].legend()

    ax[2].scatter(data_unif[:-1], data_unif[1:], s=10)
    ax[2].set_title(r"$(e^{-\tau_i}, e^{-\tau_{i+1}})$")

    x_grid = np.linspace(data.min(), data.max(), num=1000)
    ecdf = sm.distributions.ECDF(data)
    ax[1].plot(x_grid, ecdf(x_grid), label="ECDF")
    ax[1].plot(x_grid, 1 - np.exp(-x_grid), "--", label="Exact CDF")
    ax[1].set_title("Empirical CDF")
    ax[1].legend()

    print("Kolmogorov-Smirnov test p-value: ", kstest(rvs=data, cdf=lambda x: 1 - np.exp(-x)).pvalue)

    if path is not None:
        fig.savefig(fname=path, format="pdf", bbox_inches="tight", transparent=True)

    return kstest(rvs=data, cdf=lambda x: 1 - np.exp(-x)).pvalue, kstest(rvs=data, cdf=lambda x: 1 - np.exp(-x)).statistic