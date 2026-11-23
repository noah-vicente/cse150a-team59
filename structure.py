"""
Phase 3 file

What it does:
1. builds the directed graph from the gold-standard edges (all 10 genes kept),
2. finds the edges that lie on directed cycles,
3. removes the smallest set of those edges that makes the graph acyclic,
   breaking ties by dropping the edges with the lowest mutual information
   in the discretized training data,
4. lists the knockout/target pairs whose causal path is cut by the removal,
5. saves the DAG (edges, parents, topological order) as JSON and draws it.

"""
from final_project import (load_network, gene_columns, make_training_table,
                           present_goldstandard_edges, fit_tertile_bins,
                           apply_tertile_bins)
import itertools
import json
from math import comb, log
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

#Run it with: python structure.py

RESULTS_DIR = Path(__file__).resolve().parent / "results" / "structure"

# Above this many candidate edge subsets, fall back to the greedy method.
MAX_EXHAUSTIVE_SUBSETS = 200_000


# Sort key so genes order as G1, G2, ..., G10 rather than G1, G10, G2.
def gene_key(gene):
    return (len(gene), gene)


# Builds the directed gold-standard graph, keeping genes that have no edges.
def build_graph(tables):
    edges = present_goldstandard_edges(tables["goldstandard"])
    if edges is None:
        raise FileNotFoundError(
            f"Gold-standard file missing for network {tables['network_number']}."
        )

    graph = nx.DiGraph()
    graph.add_nodes_from(gene_columns(tables["wildtype"]))
    graph.add_edges_from(edges.itertuples(index=False, name=None))
    return graph


# Returns the edges that lie on at least one directed cycle.
def cycle_edges(graph):
    edges = []
    for component in nx.strongly_connected_components(graph):
        if len(component) > 1:
            edges.extend(graph.subgraph(component).edges())
    edges.extend(nx.selfloop_edges(graph))
    return sorted(edges)


# Mutual information (in nats) between two discretized gene columns.
def mutual_information(x, y):
    joint = pd.crosstab(x, y, normalize=True)
    px = joint.sum(axis=1)
    py = joint.sum(axis=0)

    total = 0.0
    for a in joint.index:
        for b in joint.columns:
            p = joint.loc[a, b]
            if p > 0:
                total += p * log(p / (px[a] * py[b]))
    return total


# Removes the given edges and reports whether the result has no cycles.
def is_acyclic_without(graph, edges_to_drop):
    trimmed = graph.copy()
    trimmed.remove_edges_from(edges_to_drop)
    return nx.is_directed_acyclic_graph(trimmed)


# Finds the smallest set of cycle edges whose removal leaves a DAG.
# Ties are broken by the lowest total mutual information, then alphabetically.
def break_cycles(graph, train_discrete):
    candidates = cycle_edges(graph)
    if not candidates:
        return graph.copy(), [], "none"

    weights = {
        (u, v): mutual_information(train_discrete[u], train_discrete[v])
        for u, v in candidates
    }

    for k in range(1, len(candidates) + 1):
        if comb(len(candidates), k) > MAX_EXHAUSTIVE_SUBSETS:
            break

        valid = [
            subset
            for subset in itertools.combinations(candidates, k)
            if is_acyclic_without(graph, subset)
        ]
        if valid:
            best = min(
                valid, key=lambda s: (sum(weights[e] for e in s), sorted(s))
            )
            dag = graph.copy()
            dag.remove_edges_from(best)
            return dag, sorted(best), "exhaustive"

    return greedy_break_cycles(graph, weights)


# Fallback for large graphs: repeatedly drop the weakest edge on some cycle.
# Not guaranteed to drop the fewest edges.
def greedy_break_cycles(graph, weights):
    dag = graph.copy()
    dropped = []
    while not nx.is_directed_acyclic_graph(dag):
        cycle = nx.find_cycle(dag)
        weakest = min(
            ((u, v) for u, v, *_ in cycle), key=lambda e: (weights[e], e)
        )
        dag.remove_edge(*weakest)
        dropped.append(weakest)
    return dag, sorted(dropped), "greedy"


# Pairs (knocked-out gene x, target y) where y is reachable from x in the
# original graph but not in the DAG. The model cannot predict these effects.
def affected_pairs(graph, dag):
    pairs = []
    for x in graph.nodes:
        lost = nx.descendants(graph, x) - nx.descendants(dag, x)
        pairs.extend((x, y) for y in lost)
    return sorted(pairs)


# Maps each gene to its sorted list of parents in the DAG.
def parents_of(dag):
    return {
        gene: sorted(dag.predecessors(gene), key=gene_key)
        for gene in sorted(dag.nodes, key=gene_key)
    }


# Writes the DAG and the cycle-breaking record for one network to JSON.
def save_structure(network_number, graph, dag, dropped, method, affected,
                   out_dir=RESULTS_DIR):
    out_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "network": network_number,
        "genes": sorted(graph.nodes, key=gene_key),
        "original_edges": sorted(graph.edges),
        "directed_cycles": sorted(sorted(c) for c in nx.simple_cycles(graph)),
        "cycle_breaking_method": method,
        "dropped_edges": dropped,
        "affected_pairs": affected,
        "edges": sorted(dag.edges),
        "parents": parents_of(dag),
        "topological_order": list(
            nx.lexicographical_topological_sort(dag, key=gene_key)
        ),
        "max_indegree": max((d for _, d in dag.in_degree()), default=0),
    }

    path = out_dir / f"network_{network_number}.json"
    path.write_text(json.dumps(record, indent=2))
    return path


# Reads a saved DAG back into networkx (for Phases 4 and 5).
def load_structure(network_number, out_dir=RESULTS_DIR):
    record = json.loads((out_dir / f"network_{network_number}.json").read_text())
    dag = nx.DiGraph()
    dag.add_nodes_from(record["genes"])
    dag.add_edges_from(tuple(e) for e in record["edges"])
    return dag, record


# Draws the network: kept edges solid, dropped edges dashed red.
def plot_structure(network_number, graph, dropped, out_dir=RESULTS_DIR):
    out_dir.mkdir(parents=True, exist_ok=True)
    genes = sorted(graph.nodes, key=gene_key)
    positions = nx.circular_layout(genes)
    kept = [e for e in graph.edges if e not in set(dropped)]

    fig, ax = plt.subplots(figsize=(6, 6))
    nx.draw_networkx_nodes(graph, positions, ax=ax, node_color="white",
                           edgecolors="black", node_size=900)
    nx.draw_networkx_labels(graph, positions, ax=ax, font_size=10)
    nx.draw_networkx_edges(graph, positions, edgelist=kept, ax=ax,
                           arrows=True, arrowsize=15, node_size=900,
                           connectionstyle="arc3,rad=0.1")
    nx.draw_networkx_edges(graph, positions, edgelist=dropped, ax=ax,
                           arrows=True, arrowsize=15, node_size=900,
                           style="dashed", edge_color="tab:red",
                           connectionstyle="arc3,rad=0.1")

    title = f"Network {network_number}"
    if dropped:
        title += " (dashed red = dropped to break cycles)"
    ax.set_title(title, fontsize=11)
    ax.axis("off")
    fig.tight_layout()

    path = out_dir / f"network_{network_number}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# Runs Phase 3 for one network and returns a summary row.
def run_structure(network_number, include_timeseries=False, out_dir=RESULTS_DIR):
    tables = load_network(network_number)
    training = make_training_table(tables, include_timeseries=include_timeseries)
    bins = fit_tertile_bins(training)
    train_discrete = apply_tertile_bins(training, bins)

    graph = build_graph(tables)
    dag, dropped, method = break_cycles(graph, train_discrete)
    affected = affected_pairs(graph, dag)

    check_structure(graph, dag, dropped)
    save_structure(network_number, graph, dag, dropped, method, affected, out_dir)
    plot_structure(network_number, graph, dropped, out_dir)

    return {
        "network": network_number,
        "edges": graph.number_of_edges(),
        "cycles": sum(1 for _ in nx.simple_cycles(graph)),
        "dropped": ", ".join(f"{u}->{v}" for u, v in dropped) or "-",
        "method": method,
        "affected_pairs": len(affected),
        "max_indegree": max((d for _, d in dag.in_degree()), default=0),
    }


# Sanity checks on the result of cycle breaking.
def check_structure(graph, dag, dropped):
    assert nx.is_directed_acyclic_graph(dag), "Result still has a cycle."
    assert set(dag.nodes) == set(graph.nodes), "Genes were lost or added."
    assert set(dag.edges) | set(dropped) == set(graph.edges), "Edges mismatch."

    on_cycles = set(cycle_edges(graph))
    for edge in dropped:
        assert edge in on_cycles, f"Dropped {edge}, which is not on a cycle."

    if not on_cycles:
        assert not dropped, "Dropped edges from an acyclic network."


if __name__ == "__main__":
    rows = [run_structure(n) for n in range(1, 6)]
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\nSaved JSON files and figures to {RESULTS_DIR}")