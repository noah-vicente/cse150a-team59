from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


DATA_ROOT = Path(__file__).resolve().parent / "DREAM4_InSilico_Size10"


# Reads tab separated table and cleans its column names.
def read_tsv(path):
    if not path.exists():
        return None

    table = pd.read_csv(path, sep="\t")
    table.columns = [str(column).strip('"') for column in table.columns]
    return table


# Loads the experiment and gold standard files for one DREAM4 network.
def load_network(network_number, data_root=DATA_ROOT):
    if network_number not in range(1, 6):
        raise ValueError("network_number must be between 1 and 5.")

    folder = data_root / f"insilico_size10_{network_number}"
    prefix = f"insilico_size10_{network_number}_"
    goldstandard_path = (
        data_root.parent
        / "DREAM4_Challenge2_GoldStandards"
        / "Size 10"
        / f"DREAM4_GoldStandard_InSilico_Size10_{network_number}.tsv"
    )

    if not folder.exists():
        raise FileNotFoundError(f"Network folder not found: {folder}")

    goldstandard = None
    if goldstandard_path.exists():
        goldstandard = pd.read_csv(
            goldstandard_path,
            sep="\t",
            header=None,
            names=["regulator", "target", "label"],
        )

    tables = {
        "wildtype": read_tsv(folder / f"{prefix}wildtype.tsv"),
        "multifactorial": read_tsv(folder / f"{prefix}multifactorial.tsv"),
        "timeseries": read_tsv(folder / f"{prefix}timeseries.tsv"),
        "knockouts": read_tsv(folder / f"{prefix}knockouts.tsv"),
        "knockdowns": read_tsv(folder / f"{prefix}knockdowns.tsv"),
        "dual_knockout_indexes": read_tsv(
            folder / f"{prefix}dualknockouts_indexes.tsv"
        ),
        "goldstandard": goldstandard,
        "network_number": network_number,
        "folder": folder,
    }

    return tables

# Returns gene columns in order and excludes the time column.
def gene_columns(table):
    if table is None:
        return []

    return [
        column
        for column in table.columns
        if column != "Time" and str(column).startswith("G")
    ]


# Combines wild type and multifactorial rows, optionally adding time-series rows.
def make_training_table(tables, include_timeseries=False):
    wildtype = tables["wildtype"]
    multifactorial = tables["multifactorial"]

    if wildtype is None or multifactorial is None:
        raise FileNotFoundError("Wild-type and multifactorial tables are required.")

    genes = gene_columns(wildtype)
    if not genes:
        raise ValueError("No gene columns were found in the wild-type table.")

    pieces = [wildtype[genes], multifactorial[genes]]

    if include_timeseries:
        timeseries = tables["timeseries"]
        if timeseries is None:
            raise FileNotFoundError("The time-series table is missing.")
        pieces.append(timeseries[genes])

    training = pd.concat(pieces, ignore_index=True)
    return training.apply(pd.to_numeric, errors="raise")


# Keeps only gold standard rows where the edge label is 1.
def present_goldstandard_edges(table):
    if table is None:
        return None

    if table.shape[1] < 3:
        raise ValueError("Expected regulator, target, and label columns.")

    present = table.loc[table["label"] == 1, ["regulator", "target"]]
    return present


# This method summarizes edge count, maximum in-degree, and cycles in one network.
def summarize_graph(tables):
    edges = present_goldstandard_edges(tables["goldstandard"])

    if edges is None:
        return {
            "network": tables["network_number"],
            "status": "gold-standard file missing",
            "edge_count": None,
            "max_indegree": None,
            "directed_cycles": None,
            "has_undirected_cycle": None,
        }

    genes = gene_columns(tables["wildtype"])
    graph = nx.DiGraph()
    graph.add_nodes_from(genes)
    graph.add_edges_from(
        edges[["regulator", "target"]].itertuples(index=False, name=None)
    )

    undirected_graph = graph.to_undirected()

    return {
        "network": tables["network_number"],
        "status": "ok",
        "edge_count": graph.number_of_edges(),
        "max_indegree": max(dict(graph.in_degree()).values(), default=0),
        "directed_cycles": list(nx.simple_cycles(graph)),
        "has_undirected_cycle": not nx.is_forest(undirected_graph),
    }


# Calculates knockout minus wild type change for every gene pair.
def knockout_effects(tables):
    wildtype = tables["wildtype"]
    knockouts = tables["knockouts"]

    if wildtype is None or knockouts is None:
        raise FileNotFoundError("Wild-type and knockout tables are required.")

    genes = gene_columns(wildtype)

    if gene_columns(knockouts) != genes:
        raise ValueError("Knockout and wild-type gene columns do not match.")

    if len(wildtype) != 1 or len(knockouts) != len(genes):
        raise ValueError(
            "Expected one wild-type row and one knockout row per gene."
        )

    effects = knockouts[genes].apply(pd.to_numeric, errors="raise")
    effects = effects.sub(wildtype[genes].iloc[0].astype(float), axis="columns")
    effects.index = genes
    effects.index.name = "knocked_out_gene"
    effects.columns.name = "affected_gene"
    return effects


# Fits per gene tertile thresholds and bin means using training rows.
def fit_tertile_bins(training):
    genes = gene_columns(training)
    thresholds = {}
    state_means = {}

    for gene in genes:
        values = pd.to_numeric(training[gene], errors="raise").dropna()
        lower, upper = values.quantile([1 / 3, 2 / 3]).to_numpy()
        thresholds[gene] = (float(lower), float(upper))

        labels = np.select(
            [values < lower, values < upper],
            ["low", "medium"],
            default="high",
        )
        grouped_means = pd.DataFrame(
            {"state": labels, "value": values.to_numpy()}
        ).groupby("state")["value"].mean()

        overall_mean = float(values.mean())
        state_means[gene] = {
            state: float(grouped_means.get(state, overall_mean))
            for state in ("low", "medium", "high")
        }

    return {"thresholds": thresholds, "state_means": state_means}


# Applies saved tertile thresholds to a table without refitting them.
def apply_tertile_bins(table, bin_parameters):
    genes = list(bin_parameters["thresholds"])
    discretized = pd.DataFrame(index=table.index)

    for gene in genes:
        lower, upper = bin_parameters["thresholds"][gene]
        values = pd.to_numeric(table[gene], errors="raise")
        discretized[gene] = np.select(
            [values < lower, values < upper],
            ["low", "medium"],
            default="high",
        )

    return discretized

# Checks that loading, summaries, knockout effects, and discretization work.
def structure_test():
    summaries = []

    for network_number in range(1, 6):
        tables = load_network(network_number)
        genes = gene_columns(tables["wildtype"])
        summary = summarize_graph(tables)

        assert len(genes) == 10
        assert tables["wildtype"].shape == (1, 10)
        assert tables["multifactorial"].shape == (10, 10)
        assert tables["timeseries"].shape == (105, 11)
        assert tables["knockouts"].shape == (10, 10)
        assert tables["goldstandard"].shape == (90, 3)
        assert summary["status"] == "ok"

        training = make_training_table(tables)
        parameters = fit_tertile_bins(training)
        discretized_training = apply_tertile_bins(training, parameters)
        discretized_knockouts = apply_tertile_bins(
            tables["knockouts"], parameters
        )
        effects = knockout_effects(tables)

        assert training.shape == (11, 10)
        assert effects.shape == (10, 10)
        assert discretized_training.shape == (11, 10)
        assert discretized_knockouts.shape == (10, 10)
        assert set(discretized_training["G1"].unique()).issubset(
            {"low", "medium", "high"}
        )

        summaries.append(summary)
        print(
            f"Network {network_number}: "
            f"{summary['edge_count']} edges; "
            f"max in-degree {summary['max_indegree']}; "
            f"{len(summary['directed_cycles'])} directed cycles"
        )

    print("\nAll five networks passed.")
    print(pd.DataFrame(summaries).to_string(index=False))


if __name__ == "__main__":
    structure_test()