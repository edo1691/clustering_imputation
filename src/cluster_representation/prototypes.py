"""Training-membership prototypes computed in original feature units."""

from typing import Any, Dict, List, Mapping

import numpy as np
import pandas as pd


def build_prototypes(
    train_features: pd.DataFrame,
    labels: Any,
    config: Mapping[str, Any],
    min_cluster_size: int,
) -> Dict[str, pd.DataFrame]:
    id_column = config["dataset"]["id_column"]
    feature_names = [feature["name"] for feature in config["features"]]
    if list(train_features.columns) != [id_column] + feature_names:
        raise ValueError("Training feature schema does not match configuration")
    cluster_labels = np.asarray(labels, dtype=int)
    if cluster_labels.ndim != 1 or len(cluster_labels) != len(train_features):
        raise ValueError("Fitted training labels do not align with training rows")
    assignments = pd.DataFrame(
        {id_column: train_features[id_column].to_numpy(), "cluster": cluster_labels}
    )
    combined = train_features[feature_names].copy()
    combined["cluster"] = cluster_labels
    grouped = combined.groupby("cluster", sort=True)
    supports = grouped.size()
    if supports.empty or (supports < min_cluster_size).any():
        raise ValueError("Fitted cluster memberships violate minimum prototype support")
    medians = grouped[feature_names].median()
    if medians.isna().any().any():
        missing = medians.columns[medians.isna().any()].tolist()
        raise ValueError("Cannot compute prototype medians for: {}".format(missing))
    prototypes = medians.reset_index()
    prototypes.insert(1, "n_clients", supports.reindex(medians.index).to_numpy())
    return {"assignments": assignments, "prototypes": prototypes}
