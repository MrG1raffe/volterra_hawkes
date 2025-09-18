import time

import numpy as np
import matplotlib.pyplot as plt
from itertools import product

from volterra_hawkes.iVi.iVi_hawkes import IVIHawkesProcess
from volterra_hawkes.utility.visualisation import get_N_U_sample, get_arrivals_sample, plot_trajectories, \
    plot_marginal_laws, poisson_jumps_test, plot_cf_convergence

from simulation.monte_carlo import MonteCarlo

def run_test0(e, path_experiment, experiment_results):
    # test 0: plot the kernel and sample trajectories
    fig, ax = plt.subplots()
    ax.plot(e.t_grid, e.kernel(e.t_grid))
    fig.savefig(path_experiment + "kernel.pdf", format="pdf", bbox_inches="tight", transparent=True)
    plot_trajectories(e, path=path_experiment + "sample_trajectories.pdf")


def run_test2(e, path_experiment, experiment_results, samples_non_ivi, n_steps_arr = (100,)):
    # test 2: marginal laws
    samples = {method: samples_non_ivi[method] for method in samples_non_ivi.keys()}
    print("Computing iVi samples...")
    ivi_methods = experiment_results["methods_ivi"]
    n_paths = 10_000
    for n_steps in n_steps_arr:
        nan_flag = False
        print("n_steps =", n_steps)
        for method in ivi_methods:
            try:
                samples[method] = get_N_U_sample(experiment=e.change_n_steps(n_steps), method=method,
                                                 n_paths=n_paths, return_counters=True)
            except Exception as ex:
                print("Exception occured:", ex)
                samples[method] = (np.array([[np.nan]]), np.array([[np.nan]]))
                nan_flag = True
        if nan_flag:
            continue
        experiment_results["U_p_values_" + str(n_steps)] = plot_marginal_laws(experiment_results, samples, "U", path=path_experiment + "marginal_laws_U_" + str(n_steps) + ".pdf")
        experiment_results["N_p_values_" + str(n_steps)] = plot_marginal_laws(experiment_results, samples, "N", path=path_experiment + "marginal_laws_N_" + str(n_steps) + ".pdf")


def run_test3(e, path_experiment, experiment_results, samples_non_ivi):
    # test 3: convergence of the CF
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
    expected_U = ivi.U_mean(t_grid=np.linspace(e.t_grid[0], e.t_grid[-1], 1000))
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
            sample = get_N_U_sample(experiment=experiment, method=method, n_paths=batch_size, rng=rng, return_counters=True)
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


def run_test4(e, path_experiment, experiment_results):
    # test 4: sampling arrivals
    methods = list(experiment_results["methods_non_ivi"]) + list(experiment_results["methods_ivi"])
    n_methods = len(methods)
    n_steps_arr = [10, 50, 100, 1000]
    for n_steps in n_steps_arr:
        rng = np.random.default_rng(seed=42)
        experiment_results["arriavals_p_values_" + str(n_steps)] = dict()
        fig, ax = plt.subplots(n_methods, 3, figsize=(12, 3*n_methods))
        nan_flag = False
        for i, method in enumerate(methods):
            try:
                samples_arrivals = get_arrivals_sample(experiment=e.change_n_steps(n_steps), method=method, n_paths=1, rng=rng)
            except Exception as ex:
                print("Exception occured:", ex)
                nan_flag = True
                continue
            print(method)
            ivi_hawkes = IVIHawkesProcess(kernel=e.kernel, g0_bar=e.g0_bar, rng=rng, g0=e.g0, resolvent_flag=False)
            experiment_results["arriavals_p_values_" + str(n_steps)][method] = poisson_jumps_test(
                ivi_hawkes.U_from_jumps(samples_arrivals[0], samples_arrivals[0]), ax=ax[i])
        if nan_flag:
            continue
        fig.savefig(path_experiment + "arrivals_" + str(n_steps) + ".pdf", format="pdf", bbox_inches="tight", transparent=True)