from dataclasses import dataclass
from typing import Callable, Union
import time

import numpy as np
from itertools import product

from notebooks.dev.visualisation import plot_marginal_laws, plot_cf_convergence

from simulation.monte_carlo import MonteCarlo
from volterra_hawkes.kernel.exponential_kernel import ExponentialKernel
from volterra_hawkes.kernel.gamma_kernel import GammaKernel
from volterra_hawkes.kernel.kernel import Kernel
from volterra_hawkes.point_processes.hawkes import simulate_hawkes_population, simulate_hawkes_ogata, U_from_jumps, N_from_jumps, simulate_exponential_hawkes
from volterra_hawkes.iVi.iVi_hawkes import IVIHawkesProcess


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

        if isinstance(self.kernel, GammaKernel):
            t0_gamma_sc = (self.kernel.alpha - 1) / self.kernel.lam
            def decreasing_kernel(t):
                return self.kernel(t) * (t > t0_gamma_sc) + self.kernel(t0_gamma_sc) * (t <= t0_gamma_sc)
            self.decreasing_kernel = decreasing_kernel

    def change_n_steps(self, n_steps):
        res = Experiment(T=self.T, n_steps=n_steps, kernel=self.kernel, mu=self.mu, decreasing_kernel=self.decreasing_kernel)
        return res


def get_N_U_sample(experiment: Experiment, method: str, n_paths: int, rng=None):
    if rng is None:
        rng = np.random.default_rng(seed=42)
    if method == "Population":
        hawkes_arrivals_sample = [simulate_hawkes_population(T=experiment.T, g0=experiment.g0, g0_upper_bound=experiment.mu,
                                                             kernel=experiment.kernel, rng=rng) for _ in range(n_paths)]
        N_sample = np.array(
            [N_from_jumps(np.array(experiment.t_grid), hawkes_arrivals) for hawkes_arrivals in hawkes_arrivals_sample])
        U_sample = np.array([U_from_jumps(np.array(experiment.t_grid), hawkes_arrivals, kernel=experiment.kernel,
                                          g0_bar=experiment.g0_bar) for hawkes_arrivals in hawkes_arrivals_sample])
        return N_sample, U_sample
    elif method == "Ogata":
        hawkes_arrivals_sample = [simulate_hawkes_ogata(T=experiment.T, mu=experiment.mu, kernel=experiment.kernel,
                                                        rng=rng, batch_size=1,
                                                        decreasing_kernel=experiment.decreasing_kernel)
                                  for _ in range(n_paths)]
        N_sample = np.array(
            [N_from_jumps(np.array(experiment.t_grid), hawkes_arrivals) for hawkes_arrivals in hawkes_arrivals_sample])
        U_sample = np.array([U_from_jumps(np.array(experiment.t_grid), hawkes_arrivals, kernel=experiment.kernel,
                                          g0_bar=experiment.g0_bar) for hawkes_arrivals in hawkes_arrivals_sample])
        return N_sample, U_sample
    elif method == "ExpExact":
        if not isinstance(experiment.kernel, ExponentialKernel):
            raise ValueError("The method ExpExact only works for exponential kernels.")
        hawkes_arrivals_sample = [simulate_exponential_hawkes(T=experiment.T, mean_intensity=experiment.mu, init_intensity=experiment.mu,
                                                              kernel=experiment.kernel, rng=rng)
                                  for _ in range(n_paths)]
        N_sample = np.array([N_from_jumps(np.array(experiment.t_grid), hawkes_arrivals)
                             for hawkes_arrivals in hawkes_arrivals_sample])
        U_sample = np.array([U_from_jumps(np.array(experiment.t_grid), hawkes_arrivals, kernel=experiment.kernel,
                                          g0_bar=experiment.g0_bar)
                             for hawkes_arrivals in hawkes_arrivals_sample])
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
            simulate_hawkes_population(T=experiment.T, g0=experiment.g0, g0_upper_bound=experiment.mu, kernel=experiment.kernel,
                                       rng=rng) for _ in range(n_paths)]
        return hawkes_arrivals_sample
    elif method == "Ogata":
        hawkes_arrivals_sample = [
            simulate_hawkes_ogata(T=experiment.T, mu=experiment.mu, kernel=experiment.kernel, rng=rng, batch_size=400,
                                  decreasing_kernel=experiment.decreasing_kernel)
            for _ in range(n_paths)]
        return hawkes_arrivals_sample
    elif method == "ExpExact":
        if not isinstance(experiment.kernel, ExponentialKernel):
            raise ValueError("The method ExpExact only works for exponential kernels.")
        hawkes_arrivals_sample = [
            simulate_exponential_hawkes(T=experiment.T, mean_intensity=experiment.mu, init_intensity=experiment.mu,
                                        kernel=experiment.kernel, rng=rng)
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
    else:
        raise ValueError(f"Method {method} not recognized.")


def test_marginal_laws(e, path_experiment, experiment_results, samples_non_ivi, n_steps_arr = (100,)):
    samples = {method: samples_non_ivi[method] for method in samples_non_ivi.keys()}
    print("Computing iVi samples...")
    ivi_methods = experiment_results["methods_ivi"]
    n_paths = 10_000
    for n_steps in n_steps_arr:
        nan_flag = False
        print("n_steps =", n_steps)
        for method in ivi_methods:
            try:
                samples[method] = get_N_U_sample(experiment=e.change_n_steps(n_steps), method=method, n_paths=n_paths)
            except Exception as ex:
                print("Exception occured:", ex)
                samples[method] = (np.array([[np.nan]]), np.array([[np.nan]]))
                nan_flag = True
        if nan_flag:
            continue
        experiment_results["U_p_values_" + str(n_steps)] = plot_marginal_laws(experiment_results, samples, "U", path=path_experiment + "marginal_laws_U_" + str(n_steps) + ".pdf")
        experiment_results["N_p_values_" + str(n_steps)] = plot_marginal_laws(experiment_results, samples, "N", path=path_experiment + "marginal_laws_N_" + str(n_steps) + ".pdf")


def test_laplace_transform(e, path_experiment, experiment_results, samples_non_ivi):
    print("Computing iVi samples...")
    is_log_time = experiment_results["is_log_time"]
    n_steps_arr = experiment_results["n_steps_arr_cf"]
    batch_size = experiment_results["batch_size_cf"]
    n_batch = experiment_results["n_batch_cf"]

    experiment_results["n_paths_ivi"] = n_batch * batch_size
    experiments = [e.change_n_steps(n_steps=n_steps) for n_steps in n_steps_arr]

    rng = np.random.default_rng(seed=42)
    ivi = IVIHawkesProcess(kernel=experiments[-1].kernel, g0_bar=experiments[-1].g0_bar, rng=rng,
                           g0=experiments[-1].g0, resolvent_flag=False)
    expected_U = ivi.get_mean(t_grid=np.linspace(e.t_grid[0], e.t_grid[-1], 1000))
    experiment_results["mean_N_T"] = expected_U[-1]
    print("Mean:", experiment_results["mean_N_T"])

    w = -1 / experiment_results["mean_N_T"]
    fun = lambda x: np.exp(w * x)
    cf_ref = {}

    for mode in ["U", "N"]:
        idx = 1 if mode == "U" else 0
        experiment_results["mc_std_" + mode] = fun(samples_non_ivi[experiment_results["methods_non_ivi"][0]][idx][:, -1]).std() / np.sqrt(experiment_results["n_paths_ivi"])
        cf_ref[mode] = ivi.characteristic_function(T=experiments[-1].T, w=w, n_steps=10000, mode=mode)


    experiments = [e.change_n_steps(n_steps=n_steps) for n_steps in n_steps_arr]
    mc_samples = dict()

    errors_ivi = {"U": {m: [] for m in experiment_results["methods_ivi"]},
                  "N": {m: [] for m in experiment_results["methods_ivi"]}}
    for method, experiment in product(experiment_results["methods_ivi"], experiments):
        print(method, experiment.n_steps)
        rng = np.random.default_rng(seed=42)
        time_ivi = time.time()
        for _ in range(n_batch):
            sample = get_N_U_sample(experiment=experiment, method=method, n_paths=batch_size, rng=rng)
            for mode in ["U", "N"]:
                idx = 1 if mode == "U" else 0
                if (method, experiment.n_steps, mode) not in mc_samples:
                    mc_samples[(method, experiment.n_steps, mode)] = MonteCarlo(batch=fun(sample[idx][:, -1]), confidence_level=0.95)
                else:
                    mc_samples[(method, experiment.n_steps, mode)].add_batch(batch=fun(sample[idx][:, -1]))
        time_ivi = time.time() - time_ivi
        experiment_results["time_" + method + "_" + str(experiment.n_steps)] = time_ivi * (experiment_results["n_paths_time_meas"] /
                                                                                           experiment_results["n_paths_ivi"])
        for mode in ["U", "N"]:
            errors_ivi[mode][method].append(np.abs(cf_ref[mode] - mc_samples[(method, experiment.n_steps, mode)].mean))
        experiment_results["errors_ivi"] = errors_ivi
    for mode in ["U", "N"]:
        for method in experiment_results["methods_ivi"]:
            experiment_results["mc_std_" + method + "_" + mode] = np.sqrt(mc_samples[(method, experiments[-1].n_steps, mode)].var / experiment_results["n_paths_ivi"])
    plot_cf_convergence(results=experiment_results, path_experiment=path_experiment, is_log_time=is_log_time)
