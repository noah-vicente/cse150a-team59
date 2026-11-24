from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pgmpy.sampling import BayesianModelSampling

from inference import (load_model, predict, mutilate,
                       probability_of_evidence, ZeroProbabilityEvidence)
from final_project import load_network, gene_columns
from metrics import total_variation

SAMPLERS = ("rejection", "likelihood_weighting", "gibbs")
EVIDENCE_SIZES = (1, 2, 3)
SAMPLE_SIZES = (1000, 5000)
MIN_REJECTION_PE = 1e-3   # skip rejection when evidence is this unlikely (it would take forever)


def sample_evidence(model, x, k, seed):
    """Pick k genes (not x) and give them realistic observed states by
    drawing one sample from the knockout model."""
    mutilated = mutilate(model, {x: "low"})
    sample = (BayesianModelSampling(mutilated)
              .forward_sample(size=1, seed=seed, show_progress=False).iloc[0])
    rng = np.random.default_rng(seed)
    candidates = [gene for gene in mutilated.nodes() if gene != x]
    chosen = rng.choice(candidates, size=k, replace=False)
    return {gene: str(sample[gene]) for gene in chosen}


def run_e3(networks=range(1, 6), knockouts=("G1", "G4", "G7"),
           evidence_sizes=EVIDENCE_SIZES, trials=5,
           sample_sizes=SAMPLE_SIZES, seeds=range(3)):
    """For each knockout and evidence size, try several random evidence sets
    and compare every sampler with exact inference. One row per run."""
    rows = []
    for network in networks:
        model = load_model(network)
        predict(model, do={"G1": "low"}, method="rejection", n_samples=10, seed=0)  # warm-up
        for x in knockouts:
            do = {x: "low"}
            for k in evidence_sizes:
                for trial in range(trials):
                    evidence = sample_evidence(model, x, k, seed=1000 * network + 100 * k + trial)
                    p_e = probability_of_evidence(model, do=do, evidence=evidence)
                    exact = predict(model, do=do, evidence=evidence, method="exact")
                    base = {"network": network, "knockout": x, "k": k, "trial": trial,
                            "evidence": str(evidence), "p_evidence": p_e}
                    rows.append({**base, "method": "exact", "n_samples": np.nan,
                                 "seed": np.nan, "tv": 0.0,
                                 "runtime": exact.runtime_seconds, "skipped": False})

                    for method in SAMPLERS:
                        for n in sample_sizes:
                            for seed in seeds:
                                row = {**base, "method": method, "n_samples": n,
                                       "seed": seed, "tv": np.nan,
                                       "runtime": np.nan, "skipped": False}
                                if method == "rejection" and p_e < MIN_REJECTION_PE:
                                    row["skipped"] = True
                                else:
                                    try:
                                        approx = predict(model, do=do, evidence=evidence,
                                                         method=method, n_samples=n, seed=seed)
                                        row["tv"] = total_variation(
                                            approx.distributions, exact.distributions,
                                            exclude=[x, *evidence])
                                        row["runtime"] = approx.runtime_seconds
                                    except ZeroProbabilityEvidence:
                                        row["skipped"] = True
                                rows.append(row)
    return pd.DataFrame(rows)


def summarize_e3(runs):
    """Average error and runtime per sampler, evidence size, and sample size."""
    samplers = runs[runs["method"] != "exact"]
    return (samplers.groupby(["method", "k", "n_samples"])
            .agg(tv_mean=("tv", "mean"), runtime_mean=("runtime", "mean"),
                 p_evidence_mean=("p_evidence", "mean"), skipped=("skipped", "sum"))
            .reset_index())


def plot_e3(runs, n=5000, path="results/e3_evidence.png"):
    """Left: error vs P(evidence). Right: runtime vs P(evidence)."""
    data = runs[(runs["method"] != "exact") & (runs["n_samples"] == n)]
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4))
    for method, group in data.groupby("method"):
        left.scatter(group["p_evidence"], group["tv"], s=12, alpha=0.6, label=method)
        right.scatter(group["p_evidence"], group["runtime"], s=12, alpha=0.6, label=method)
    for ax, ylabel in ((left, "total variation vs exact"), (right, "runtime (seconds)")):
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("P(evidence | do)"); ax.set_ylabel(ylabel)
        ax.legend()
    fig.suptitle(f"E3: queries with evidence (n = {n} samples)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)


if __name__ == "__main__":
    Path("results").mkdir(exist_ok=True)
    runs = run_e3()   # defaults: 5 networks, 3 knockouts, 5 trials, 3 seeds
    runs.to_csv("results/e3_runs.csv", index=False)
    print(summarize_e3(runs).round(4).to_string(index=False))
    plot_e3(runs)