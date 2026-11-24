import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score   # only if the group adds scikit-learn

from inference import predict

def clean(series, tol=1e-9):
    """Treat floating-point leftovers (|x| < tol) as exactly zero."""
    return series.where(series.abs() > tol, 0.0)

def expected_values(distributions, state_means):
    """Turn per-gene state probabilities into one expected expression value per gene.

    distributions: DataFrame from predict(...).distributions
                   (columns: gene, low, medium, high, predicted_state)
    state_means:   {gene: {"low": mean, "medium": mean, "high": mean}}
    Returns:       pd.Series of E[Y], indexed by gene name
    """
    values = {}
    for _, row in distributions.iterrows():
        gene = row["gene"]
        means = state_means[gene]
        values[gene] = sum(row[state] * means[state] for state in ("low", "medium", "high"))
    return pd.Series(values)

def predicted_effects(model, x, state_means, observational=False, method="exact", **predict_kwargs):
    """Predicted change in every other gene when gene x is set to 'low'.

    observational=False: E[Y | do(X=low)] - E[Y]   (the knockout, main prediction)
    observational=True:  E[Y | X=low]     - E[Y]   (plain conditioning, baseline 3)
    Returns a pd.Series indexed by gene, with x itself left out.
    """
    before = predict(model, method=method, **predict_kwargs).distributions

    if observational:
        after = predict(model, evidence={x: "low"}, method=method, **predict_kwargs).distributions
    else:
        after = predict(model, do={x: "low"}, method=method, **predict_kwargs).distributions

    effects = expected_values(after, state_means) - expected_values(before, state_means)
    return effects.drop(x)

def direction_accuracy(predicted, observed, threshold=0.1):
    """Fraction of gene pairs where the predicted direction (up/down) matches reality.

    predicted: pd.Series of predicted effects, indexed by gene
    observed:  pd.Series of real effects (KO - WT), indexed by gene
    threshold: only count genes whose real change is bigger than this,
               since tiny changes are mostly noise and have no meaningful direction
    Returns:   a fraction between 0 and 1 (NaN if no gene passes the threshold)
    """
    predicted, observed = predicted.align(observed, join="inner")
    predicted = clean(predicted)
    changed = observed.abs() > threshold
    if changed.sum() == 0:
        return np.nan
    matches = np.sign(predicted[changed]) == np.sign(observed[changed])
    return matches.mean()

def effect_correlation(predicted, observed):
    """Spearman rank correlation between predicted and real effects.

    +1 = the model ranks effects in exactly the right order
     0 = no relationship
    -1 = exactly backwards
    Returns NaN if there are too few values or one side is constant.
    """
    predicted, observed = predicted.align(observed, join="inner")
    predicted = clean(predicted)
    if len(predicted) < 3 or predicted.nunique() < 2 or observed.nunique() < 2:
        return np.nan
    rho, _ = spearmanr(predicted, observed)
    return rho

def state_accuracy(distributions, true_states, x=None):
    """Fraction of genes whose most likely predicted state matches the real state.

    distributions: DataFrame from predict(...).distributions (has 'gene' and 'predicted_state')
    true_states:   pd.Series of real discretized states ('low'/'medium'/'high'), indexed by gene
    x:             the knocked-out gene, left out of scoring
    """
    predicted = distributions.set_index("gene")["predicted_state"]
    if x is not None:
        predicted = predicted.drop(x)
    predicted, true_states = predicted.align(true_states, join="inner")
    return (predicted == true_states).mean()

def affected_auroc(predicted, observed, threshold=0.1):
    """How well |predicted effect| separates really-affected genes from unaffected ones.

    1.0 = every affected gene got a bigger predicted effect than every unaffected gene
    0.5 = no better than random guessing
    Returns NaN if all genes are affected or all are unaffected (AUROC needs both).
    """
    predicted, observed = predicted.align(observed, join="inner")
    predicted = clean(predicted)
    labels = (observed.abs() > threshold).astype(int)
    if labels.nunique() < 2:
        return np.nan
    scores = predicted.abs()
    return roc_auc_score(labels, scores)

def total_variation(approx_distributions, exact_distributions, exclude=()):
    """Average total variation distance between approximate and exact marginals.

    For each gene: TV = 0.5 * (|p_low - q_low| + |p_med - q_med| + |p_high - q_high|)
    0 = identical distributions, 1 = completely different.
    exclude: genes to skip, such as the knocked-out gene and any evidence genes,
             which are fixed in both answers and would always score 0.
    """
    states = ["low", "medium", "high"]
    approx = approx_distributions.set_index("gene")[states]
    exact = exact_distributions.set_index("gene")[states]
    approx, exact = approx.align(exact, join="inner")
    tv = 0.5 * (approx - exact).abs().sum(axis=1)
    return tv.drop(list(exclude), errors="ignore").mean()

def baseline_no_change(genes, x):
    """Baseline 1: predict that knocking out x changes nothing.
    Returns a predicted effect of 0 for every gene except x."""
    return pd.Series(0.0, index=[gene for gene in genes if gene != x])

def baseline_graph_only(graph, x, genes):
    """Baseline 2: predict 'affected' (1) for every gene downstream of x
    in the gold-standard graph, and 'not affected' (0) for everything else."""
    downstream = nx.descendants(graph, x)
    return pd.Series(
        [1.0 if gene in downstream else 0.0 for gene in genes if gene != x],
        index=[gene for gene in genes if gene != x],
    )

if __name__ == "__main__":
    # Sanity checks with hand-computed answers
    dist = pd.DataFrame([{"gene": "G1", "low": 0.6, "medium": 0.3, "high": 0.1}])
    means = {"G1": {"low": 0.2, "medium": 0.5, "high": 0.8}}
    assert abs(expected_values(dist, means)["G1"] - 0.35) < 1e-9

    pred = pd.Series({"G2": -0.2, "G3": 0.1, "G4": 0.0, "G5": 0.3})
    obs = pd.Series({"G2": -0.3, "G3": -0.2, "G4": 0.4, "G5": 0.01})
    assert abs(direction_accuracy(pred, obs, 0.1) - 1 / 3) < 1e-9

    pred = pd.Series({"G2": 0.01, "G3": 0.05, "G4": 0.02, "G5": 0.10})
    obs = pd.Series({"G2": 0.10, "G3": 0.40, "G4": 0.20, "G5": 0.90})
    assert abs(effect_correlation(pred, obs) - 1.0) < 1e-9

    pred = pd.Series({"G2": 0.30, "G3": 0.20, "G4": 0.01, "G5": 0.02})
    obs = pd.Series({"G2": -0.5, "G3": 0.4, "G4": 0.02, "G5": -0.03})
    assert affected_auroc(pred, obs, 0.1) == 1.0

    print("All metric checks passed.")