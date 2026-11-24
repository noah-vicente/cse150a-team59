import pandas as pd
import networkx as nx

from inference import load_model, predict, ZeroProbabilityEvidence
from final_project import (
    load_network, make_training_table, fit_tertile_bins,
    apply_tertile_bins, gene_columns, knockout_effects,
    present_goldstandard_edges,
)
from metrics import predicted_effects, baseline_graph_only

from pathlib import Path
import numpy as np
from metrics import direction_accuracy, effect_correlation, affected_auroc

import json
from inference import MODEL_DIR


def load_bins(network):
    """Bins saved by learn.py, so predictions use exactly the model's discretization."""
    with open(MODEL_DIR / f"network_{network}_bins.json") as f:
        return json.load(f)

def collect_pairs(network, training_set="baseline", estimator="interpolated"):
    """One row per (knocked-out gene, target gene) pair in one network,
    with the model's predictions, the baselines, and the real outcome."""
    # --- Load everything for this network once ---
    tables = load_network(network)
    model = load_model(network, training_set=training_set, estimator=estimator)
    genes = gene_columns(tables["wildtype"])

    params = load_bins(network)
    state_means = params["state_means"]

    observed = knockout_effects(tables)                     # real KO - WT, rows = knocked-out gene
    true_states = apply_tertile_bins(tables["knockouts"], params)
    true_states.index = genes                               # label rows G1..G10
    wt_states = apply_tertile_bins(tables["wildtype"], params).iloc[0]

    edges = present_goldstandard_edges(tables["goldstandard"])
    graph = nx.DiGraph(list(edges.itertuples(index=False, name=None)))
    graph.add_nodes_from(genes)                             # include genes with no edges

    # --- Loop over every knockout, then every other gene ---
    rows = []
    for x in genes:
        do_effect = predicted_effects(model, x, state_means)
        try:
            obs_effect = predicted_effects(model, x, state_means, observational=True)
        except ZeroProbabilityEvidence:
            obs_effect = pd.Series(np.nan, index=do_effect.index)
        graph_score = baseline_graph_only(graph, x, genes)
        do_probs = (predict(model, do={x: "low"}).distributions
                    .set_index("gene")[["low", "medium", "high"]].round(9))
        do_states = do_probs.idxmax(axis=1)   # ties always go to the first state (low < medium < high)

        for y in genes:
            if y == x:
                continue
            rows.append({
                "network": network,
                "knockout": x,
                "target": y,
                "observed_effect": observed.loc[x, y],
                "do_effect": do_effect[y],
                "observational_effect": obs_effect[y],
                "graph_score": graph_score[y],
                "no_change_effect": 0.0,
                "predicted_state": do_states[y],
                "true_state": true_states.loc[x, y],
                "wildtype_state": wt_states[y],
            })

    return pd.DataFrame(rows)

# Which column holds each method's predicted effect
METHODS = {
    "model (do)": "do_effect",
    "observational": "observational_effect",
    "graph only": "graph_score",
    "no change": "no_change_effect",
}


def score_group(group, threshold):
    """Compute every metric for every method on one set of pairs."""
    observed = group["observed_effect"]
    rows = []
    for method, column in METHODS.items():
        valid = group[column].notna()          # skip pairs this method couldn't answer
        predicted = group.loc[valid, column]
        observed_valid = observed[valid]
        is_graph = column == "graph_score"     # graph-only has no direction or size
        rows.append({
            "method": method,
            "n_scored": int(valid.sum()),
            "direction_accuracy": np.nan if is_graph else direction_accuracy(predicted, observed_valid, threshold),
            "spearman": np.nan if is_graph else effect_correlation(predicted, observed_valid),
            "auroc": affected_auroc(predicted, observed_valid, threshold),
        })

    # State accuracy only exists for the model and the no-change baseline
    state = {
        "model (do)": (group["predicted_state"] == group["true_state"]).mean(),
        "no change": (group["wildtype_state"] == group["true_state"]).mean(),
    }
    for row in rows:
        row["state_accuracy"] = state.get(row["method"], np.nan)
    return rows


def summarize(pairs, threshold=0.1):
    """One row per (network, method) with every metric, plus pooled 'all' rows."""
    results = []
    for network, group in pairs.groupby("network"):
        for row in score_group(group, threshold):
            results.append({"network": network, **row})
    for row in score_group(pairs, threshold):          # all 450 pairs together
        results.append({"network": "all", **row})
    return pd.DataFrame(results)


if __name__ == "__main__":
    pairs = pd.concat([collect_pairs(n) for n in range(1, 6)], ignore_index=True)
    summary = summarize(pairs, threshold=0.1)

    Path("results").mkdir(exist_ok=True)
    pairs.to_csv("results/e1_pairs.csv", index=False)
    summary.to_csv("results/e1_summary.csv", index=False)

    print(summary.round(3).to_string(index=False))