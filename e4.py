from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from final_project import load_network, present_goldstandard_edges, gene_columns
from metrics import direction_accuracy, effect_correlation

PAIRS_PATH = "results/e1_pairs.csv"


def gold_graph(network):
    """True wiring for one network, including feedback loops."""
    tables = load_network(network)
    edges = present_goldstandard_edges(tables["goldstandard"])
    graph = nx.DiGraph(list(edges.itertuples(index=False, name=None)))
    graph.add_nodes_from(gene_columns(tables["wildtype"]))
    return graph


def add_relationship(pairs):
    """Label each (knockout, target) pair by how target relates to knockout
    in the true wiring: downstream, upstream, both (feedback loop), or unrelated."""
    labels = []
    graphs = {n: gold_graph(n) for n in pairs["network"].unique()}
    for _, row in pairs.iterrows():
        graph = graphs[row["network"]]
        down = row["target"] in nx.descendants(graph, row["knockout"])
        up = row["target"] in nx.ancestors(graph, row["knockout"])
        labels.append("both" if down and up else
                      "downstream" if down else
                      "upstream" if up else "unrelated")
    return pairs.assign(relationship=labels)


def summarize_e4(pairs, threshold=0.1):
    """Compare do vs observational within each relationship group."""
    rows = []
    groups = list(pairs.groupby("relationship")) + [("all", pairs)]
    for relationship, group in groups:
        observed = group["observed_effect"]
        row = {"relationship": relationship,
               "n_pairs": len(group),
               "really_changed": (observed.abs() > threshold).mean()}
        for name, column in (("do", "do_effect"), ("obs", "observational_effect")):
            predicted = group[column]
            row[f"{name}_nonzero"] = (predicted.abs() > 1e-9).mean()
            row[f"{name}_direction"] = direction_accuracy(predicted, observed, threshold)
            row[f"{name}_spearman"] = effect_correlation(predicted, observed)
        rows.append(row)
    return pd.DataFrame(rows)


def disagreements(pairs, threshold=0.1):
    """Among pairs where do and observational predict different effects and the
    gene really changed, how often does each get the direction right?"""
    differ = (pairs["do_effect"] - pairs["observational_effect"]).abs() > 1e-9
    changed = pairs["observed_effect"].abs() > threshold
    subset = pairs[differ & changed]
    true_sign = np.sign(subset["observed_effect"])
    return {
        "pairs_where_they_differ_and_gene_changed": len(subset),
        "do_right": (np.sign(subset["do_effect"]) == true_sign).mean(),
        "obs_right": (np.sign(subset["observational_effect"]) == true_sign).mean(),
    }


def plot_e4(summary, path="results/e4_do_vs_obs.png"):
    """Direction accuracy of do vs observational in each relationship group."""
    data = summary.set_index("relationship")[["do_direction", "obs_direction"]]
    ax = data.plot(kind="bar", figsize=(7, 4), rot=0)
    ax.set_ylabel("direction accuracy")
    ax.set_title("E4: intervening vs observing, by relationship to the knocked-out gene")
    ax.legend(["do(X = low)", "X = low (observed)"])
    plt.tight_layout()
    plt.savefig(path, dpi=150)


if __name__ == "__main__":
    pairs = add_relationship(pd.read_csv(PAIRS_PATH))
    summary = summarize_e4(pairs, threshold=0.1)
    pairs.to_csv("results/e4_pairs.csv", index=False)
    summary.to_csv("results/e4_summary.csv", index=False)

    print(summary.round(3).to_string(index=False))
    print("\n", disagreements(pairs))
    plot_e4(summary)