import json
from collections import Counter
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from pgmpy.models import DiscreteBayesianNetwork

from final_project import (
    apply_tertile_bins,
    fit_tertile_bins,
    load_network,
    make_training_table,
)
from structure import RESULTS_DIR as STRUCTURE_DIR
from structure import load_structure, parents_of, run_structure

RESULTS_DIR = Path(__file__).resolve().parent / "results" / "models"
STATES = ["low", "medium", "high"]
UNIFORM_WEIGHTS = {"mle": 0.0, "interpolated": 0.25}


def fit_models(dag, training):
    models = {}
    for estimator in UNIFORM_WEIGHTS:
        model = DiscreteBayesianNetwork(dag.edges())
        model.add_nodes_from(dag.nodes())
        models[estimator] = model

    coverage = []
    for gene, parents in parents_of(dag).items():
        configurations = list(product(STATES, repeat=len(parents)))
        counts = Counter(training[[*parents, gene]].itertuples(index=False, name=None))
        counts_by_state = np.array(
            [[counts[(*config, state)] for config in configurations] for state in STATES],
            dtype=float,
        )
        totals = counts_by_state.sum(axis=0)
        unseen = totals == 0
        coverage.append({
            "gene": gene,
            "parent_configurations": len(configurations),
            "unseen_configurations": int(unseen.sum()),
            "unseen_fraction": float(unseen.mean()),
        })

        # An unseen parent configuration has no MLE, so use a uniform row.
        mle = np.full_like(counts_by_state, 1 / len(STATES))
        np.divide(counts_by_state, totals, out=mle, where=~unseen)

        for estimator, weight in UNIFORM_WEIGHTS.items():
            cpt = TabularCPD(
                variable=gene,
                variable_card=len(STATES),
                values=(1 - weight) * mle + weight / len(STATES),
                evidence=parents,
                evidence_card=[len(STATES)] * len(parents),
                state_names={node: STATES for node in [gene, *parents]},
            )
            models[estimator].add_cpds(cpt)

    return models, coverage


def wildtype_mismatches(model, wildtype):
    inference = VariableElimination(model)
    mismatches = []
    for gene, state in wildtype.items():
        marginal = inference.query([gene], show_progress=False)
        if not np.isclose(marginal.get_value(**{gene: state}), marginal.values.max()):
            mismatches.append(gene)
    return mismatches


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary = []
    coverage = []

    for network in range(1, 6):
        if not (STRUCTURE_DIR / f"network_{network}.json").exists():
            run_structure(network)
        dag, _ = load_structure(network)

        tables = load_network(network)
        baseline = make_training_table(tables)
        # Use the same bins in both fits so the training rows are the only change.
        bins = fit_tertile_bins(baseline)
        (RESULTS_DIR / f"network_{network}_bins.json").write_text(
            json.dumps(bins, indent=2) + "\n"
        )
        wildtype = apply_tertile_bins(tables["wildtype"], bins).iloc[0]
        augmented = make_training_table(tables, include_timeseries=True)

        for training_set, training in (("baseline", baseline), ("with_timeseries", augmented)):
            models, gene_coverage = fit_models(dag, apply_tertile_bins(training, bins))
            info = {"network": network, "training_set": training_set, "training_rows": len(training)}
            coverage.extend({**info, **row} for row in gene_coverage)
            total = sum(row["parent_configurations"] for row in gene_coverage)
            unseen = sum(row["unseen_configurations"] for row in gene_coverage)

            for estimator, model in models.items():
                model.name = f"network_{network}_{training_set}_{estimator}"
                filename = f"{model.name}.bif"
                mismatches = wildtype_mismatches(model, wildtype)
                model.save(str(RESULTS_DIR / filename), filetype="bif")
                summary.append({
                    **info,
                    "estimator": estimator,
                    "uniform_weight": UNIFORM_WEIGHTS[estimator],
                    "parent_configurations": total,
                    "unseen_configurations": unseen,
                    "unseen_fraction": unseen / total,
                    "wildtype_matches": len(wildtype) - len(mismatches),
                    "wildtype_mismatches": ",".join(mismatches),
                    "model_file": filename,
                })

    pd.DataFrame(summary).to_csv(RESULTS_DIR / "summary.csv", index=False)
    pd.DataFrame(coverage).to_csv(RESULTS_DIR / "coverage.csv", index=False)
    print(f"Saved {len(summary)} models and diagnostics to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
