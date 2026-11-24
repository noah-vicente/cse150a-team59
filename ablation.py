from pathlib import Path
import pandas as pd

from e1 import collect_pairs, summarize

# One entry per model version: (training_set, estimator)
# Matches the filenames in results/models
CONFIGS = [
    ("baseline", "mle"),
    ("baseline", "interpolated"),
    ("with_timeseries", "mle"),
    ("with_timeseries", "interpolated"),
]


def run_ablation(configs=CONFIGS, threshold=0.1):
    summaries = []
    for training_set, estimator in configs:
        print(f"running {training_set} / {estimator}", flush=True)
        pairs = pd.concat(
            [collect_pairs(n, training_set, estimator) for n in range(1, 6)],
            ignore_index=True,
        )
        pairs.to_csv(f"results/ablation_pairs_{training_set}_{estimator}.csv", index=False)
        summary = summarize(pairs, threshold)
        summary.insert(0, "config", f"{training_set} / {estimator}")
        summaries.append(summary)
    return pd.concat(summaries, ignore_index=True)


if __name__ == "__main__":
    Path("results").mkdir(exist_ok=True)
    results = run_ablation()
    results.to_csv("results/ablation_summary.csv", index=False)

    headline = results[results["network"] == "all"]
    print(headline.round(3).to_string(index=False))