from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from inference import load_model, predict
from final_project import load_network, gene_columns
from metrics import total_variation

SAMPLERS = ("rejection", "likelihood_weighting", "gibbs")
SAMPLE_SIZES = (100, 300, 1000, 3000, 5000, 10000)
SEEDS = range(10)


def run_e2(networks=range(1, 6), knockouts=None,
           sample_sizes=SAMPLE_SIZES, seeds=SEEDS):
    """For every knockout, compare each sampler with exact inference
    across sample sizes and seeds. One row per run."""
    rows = []
    for network in networks:
        model = load_model(network)
        predict(model, do={"G1": "low"}, method="rejection", n_samples=10, seed=0)  # warm-up, not recorded
        genes = gene_columns(load_network(network)["wildtype"])
        for x in (knockouts or genes):
            exact = predict(model, do={x: "low"}, method="exact")
            rows.append({"network": network, "knockout": x, "method": "exact",
                         "n_samples": np.nan, "seed": np.nan, "tv": 0.0,
                         "runtime": exact.runtime_seconds})
            for method in SAMPLERS:
                for n in sample_sizes:
                    for seed in seeds:
                        approx = predict(model, do={x: "low"}, method=method,
                                         n_samples=n, seed=seed)
                        rows.append({
                            "network": network, "knockout": x, "method": method,
                            "n_samples": n, "seed": seed,
                            "tv": total_variation(approx.distributions,
                                                  exact.distributions, exclude=[x]),
                            "runtime": approx.runtime_seconds,
                        })
    return pd.DataFrame(rows)


def summarize_e2(runs, target_tv=0.01):
    """Average error and runtime per method and sample size,
    plus the smallest sample size that reaches the target error."""
    samplers = runs[runs["method"] != "exact"]
    summary = (samplers.groupby(["method", "n_samples"])
               .agg(tv_mean=("tv", "mean"), tv_std=("tv", "std"),
                    runtime_mean=("runtime", "mean"))
               .reset_index())

    needed = {}
    for method, group in summary.groupby("method"):
        good = group[group["tv_mean"] < target_tv]
        needed[method] = good["n_samples"].min() if len(good) else np.nan
    exact_time = runs.loc[runs["method"] == "exact", "runtime"].mean()
    return summary, needed, exact_time


def plot_e2(summary, exact_time, path="results/e2_convergence.png"):
    """Left: error vs number of samples. Right: error vs runtime."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4))
    for method, group in summary.groupby("method"):
        left.errorbar(group["n_samples"], group["tv_mean"], yerr=group["tv_std"],
                      marker="o", capsize=3, label=method)
        right.plot(group["runtime_mean"], group["tv_mean"], marker="o", label=method)
    right.axvline(exact_time, linestyle="--", color="gray", label="exact runtime")
    for ax, xlabel in ((left, "number of samples"), (right, "runtime (seconds)")):
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(xlabel); ax.set_ylabel("total variation vs exact")
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)

if __name__ == "__main__":
    Path("results").mkdir(exist_ok=True)
    runs = run_e2(networks=range(1, 6), knockouts=["G1", "G4", "G7"], seeds=range(10))
    runs.to_csv("results/e2_runs.csv", index=False)

    summary, needed, exact_time = summarize_e2(runs)
    print(summary.round(4).to_string(index=False))
    print("\nsamples needed for TV < 0.01:", needed)
    print("exact runtime (s):", round(exact_time, 4))
    plot_e2(summary, exact_time)