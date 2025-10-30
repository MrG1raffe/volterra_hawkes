import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from statsmodels.graphics.gofplots import qqplot_2samples
from scipy.stats import ks_2samp, kstest
from scipy.stats import expon
import statsmodels.api as sm

from volterra_hawkes.iVi.iVi_hawkes import IVIHawkesProcess


DEFAULT_COLOR_CYCLE = ("#B56246", "#579F40", "#9A46B5", "#4699B5", "#B54662", "#D4A017", "#5b6c64", "#71074E", "#0033A0")

def get_process_name(mode: str):
    return "$\Lambda_T$" if mode == "U" else "$N_T$"


def plot_trajectories(e, n_paths=10, path: str = None, resolvent_flag: bool = False):
    rng = np.random.default_rng(seed=42)

    ivi_hawkes = IVIHawkesProcess(kernel=e.kernel, g0_bar=e.g0_bar, g0_bar_res=e.g0_bar_res, rng=rng, g0=e.g0,
                                  resolvent_flag=resolvent_flag)

    N, U, lam = ivi_hawkes.simulate_on_grid(t_grid=e.t_grid, n_paths=n_paths)

    fig, ax = plt.subplots(1, 3, figsize=(13, 3))
    n_show = n_paths

    ax[0].plot(e.t_grid, N[:, 0:n_show])
    ax[0].set_xlabel('t')
    ax[0].set_ylabel(r'Hawkes process $N$')
    ax[0].grid("on")

    ax[1].plot(e.t_grid, U[:, 0:n_show])
    ax[1].set_xlabel('t')
    ax[1].set_ylabel(r'Integrated intensity $\Lambda$')
    ax[1].grid("on")

    ax[2].plot(e.t_grid, lam[:, 0:n_show])
    ax[2].set_xlabel('t')
    ax[2].set_ylabel(r'Instantaneous intensity $\lambda$')
    ax[2].grid("on")

    if path is not None:
        fig.savefig(fname=path, format="pdf", bbox_inches="tight", transparent=True)


def plot_marginal_laws(results, samples, flag, path=None):
    methods = results["methods_non_ivi"] + results["methods_ivi"][0:1]
    exact_method = methods[0]

    if flag == "N":
        idx = 0
    elif flag == "U":
        idx = 1
    else:
        raise ValueError("flag must be 'N' or 'U'")

    fig, ax = plt.subplots(2, 2, figsize=(12, 6))

    X_T_exact = samples[exact_method][idx][:, -1]
    x_grid = np.linspace(0, np.max(X_T_exact), 500)
    if flag == "N":
        bins = np.arange(np.floor(np.max(X_T_exact)))
    else:
        bins = x_grid[::10]

    ecdf_exact = sm.distributions.ECDF(samples[exact_method][idx][:, -1])
    ax[0, 0].hist(samples[exact_method][idx][:, -1], density=True, bins=bins, alpha=0.3, color="k", label=methods[0])
    ax[0, 0].set_ylabel(r"Histogram of " + get_process_name(flag))
    ax[0, 1].set_ylabel(r"ECDF distance, " + get_process_name(flag))

    p_values_dict = dict()
    for method, ax_qq in zip(methods[1:], [ax[1, 0], ax[1, 1]]):
        X_T = samples[method][idx][:, -1]
        if flag == "N":
            X_T = np.round(X_T)
        ecdf = sm.distributions.ECDF(X_T)
        ax[0, 0].hist(X_T, density=True, bins=bins, alpha=0.3, label=method)
        ax[0, 1].plot(x_grid, np.abs(ecdf(x_grid) - ecdf_exact(x_grid)), label=method)
        qqplot_2samples(sm.ProbPlot(X_T_exact), sm.ProbPlot(X_T), ax=ax_qq, xlabel=exact_method, ylabel=method,
                        line="45")

        print(f"p-value {exact_method}-{method}:", ks_2samp(X_T, X_T_exact).pvalue)
        p_values_dict[f"{exact_method}-{method}"] = ks_2samp(X_T, X_T_exact).pvalue, ks_2samp(X_T, X_T_exact).statistic

    ax[0, 0].legend()
    ax[0, 1].legend()

    if path is not None:
        fig.savefig(fname=path, format="pdf", bbox_inches="tight", transparent=True)


def poisson_jumps_test(jumps, path=None,
                       color_cycle=DEFAULT_COLOR_CYCLE,
                       ax: matplotlib.axes = None, fig = None):
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
    ax[0].scatter(theoretical_quantiles, empirical_quantiles, s=10)
    ax[0].plot(theoretical_quantiles, theoretical_quantiles, c=color_cycle[1], linestyle="dashed")

    ax[0].set_xlabel("Theoretical Quantiles (Exponential)")
    ax[0].set_ylabel("Empirical Quantiles (Data)")

    ax[2].scatter(data_unif[:-1], data_unif[1:], s=10)
    ax[2].set_xlabel(r"$e^{-\tau_i}$")
    ax[2].set_ylabel(r"$e^{-\tau_{i+1}}$")

    x_grid = np.linspace(data.min(), data.max(), num=1000)
    ecdf = sm.distributions.ECDF(data)
    ax[1].plot(x_grid, ecdf(x_grid), label="ECDF")
    ax[1].plot(x_grid, 1 - np.exp(-x_grid), "--", label="Exact")
    ax[1].set_ylabel("CDF")
    ax[1].legend()

    print("Kolmogorov-Smirnov test p-value: ", kstest(rvs=data, cdf=lambda x: 1 - np.exp(-x)).pvalue)

    if path is not None:
        fig.savefig(fname=path, format="pdf", bbox_inches="tight", transparent=True)

    return kstest(rvs=data, cdf=lambda x: 1 - np.exp(-x)).pvalue, kstest(rvs=data, cdf=lambda x: 1 - np.exp(-x)).statistic


def plot_cf_convergence(results: dict, path_experiment: str, is_log_time: bool = False):
    n_steps_arr = results["n_steps_arr_cf"]
    methods_ivi = results["methods_ivi"]

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for mode in ["U", "N"]:
        idx = 1 if mode == "U" else 0
        for method in methods_ivi:
            ax[idx].loglog(n_steps_arr, results["errors_ivi"][mode][method], label=method)
        ax[idx].hlines(y=3 * results["mc_std_" + mode], xmin=n_steps_arr[0], xmax=n_steps_arr[-1], color="k", linestyles="--", label="3 std")
        ax[idx].set_xlabel(r"$\log(n)$")
        process_name = "$\Lambda_T$" if mode == "U" else "$N_T$"
        ax[idx].set_ylabel(r"Absolute error, " + process_name)
        ax[idx].legend()
    fig.savefig(path_experiment + "CF_convergence.pdf", format="pdf", bbox_inches="tight", transparent=True)


    fig, ax_arr = plt.subplots(1, 2, figsize=(12, 4))
    methods_non_ivi = results["methods_non_ivi"]
    methods_ivi = results["methods_ivi"]
    for mode in ["U", "N"]:
        idx = 1 if mode == "U" else 0
        ax = ax_arr[idx]
        for method in methods_ivi:
            exec_times = [results["time_" + method + "_" + str(n_steps)] for n_steps in n_steps_arr]
            ax.scatter(results["errors_ivi"][mode][method], exec_times, marker="x", label=method)

        for method, color in zip(methods_non_ivi, DEFAULT_COLOR_CYCLE[3:]):
            ax.hlines(y=results["time_" + method], xmin=0, xmax=np.max(results["errors_ivi"][mode]["iVi"]),
                      color=color, label=method)

        ymax = max([results["time_" + method] for method in methods_non_ivi])
        ax.vlines(x=3 * results["mc_std_" + mode], ymin=0, ymax=ymax, color="k", linestyles="--", label="3 std")

        ax.legend()
        if is_log_time:
            ax.set_yscale('log')
        ax.set_xscale('log')
        process_name = "$\Lambda_T$" if mode == "U" else "$N_T$"
        ax.set_xlabel(f"Absolute error, " + process_name)
        ax.set_ylabel("Simulations time, s")
        ax.legend()
    fig.savefig(path_experiment + "CF_convergence_exec_time.pdf", format="pdf", bbox_inches="tight", transparent=True)