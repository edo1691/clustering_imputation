"""Test cluster assignment and cluster-median feature replacement."""

from typing import Any, Dict, Mapping

import numpy as np
import pandas as pd


def assign_test_clusters(
    test_features: pd.DataFrame, config: Mapping[str, Any], preprocessor: Any, model: Any
) -> Dict[str, Any]:
    id_column = config["dataset"]["id_column"]
    feature_names = [feature["name"] for feature in config["features"]]
    _require_columns(test_features, [id_column] + feature_names, "test features")
    if test_features[id_column].duplicated().any():
        raise ValueError("Test feature IDs must be unique")
    model_inputs = preprocessor.transform(test_features[feature_names])
    assignments = pd.DataFrame(
        {
            id_column: test_features[id_column].to_numpy(),
            "cluster": model.predict(model_inputs),
        }
    )
    return {"assignments": assignments, "model_inputs": model_inputs}


def transform_test(
    test_features: pd.DataFrame,
    indicator_frame: pd.DataFrame,
    prototypes: pd.DataFrame,
    assignments: pd.DataFrame,
    config: Mapping[str, Any],
    indicator_name: str,
) -> Dict[str, pd.DataFrame]:
    id_column = config["dataset"]["id_column"]
    feature_names = [feature["name"] for feature in config["features"]]
    _require_columns(test_features, [id_column] + feature_names, "test features")
    _require_columns(indicator_frame, [id_column, indicator_name], "test indicators")
    _require_columns(assignments, [id_column, "cluster"], "test assignments")
    _require_columns(prototypes, ["cluster", "n_clients"] + feature_names, "prototypes")
    for name, frame in (
        ("test features", test_features),
        ("test indicators", indicator_frame),
        ("test assignments", assignments),
        ("prototypes", prototypes),
    ):
        if id_column in frame and frame[id_column].duplicated().any():
            raise ValueError("{} contains duplicate IDs".format(name))
    expected_ids = test_features[id_column].tolist()
    if indicator_frame[id_column].tolist() != expected_ids:
        raise ValueError("Indicator IDs/order do not match test features")
    if assignments[id_column].tolist() != expected_ids:
        raise ValueError("Predicted assignment IDs/order do not match test features")
    values = indicator_frame[indicator_name]
    if values.isna().any() or not values.isin([0, 1]).all():
        raise ValueError("{} must contain only binary 0/1 values".format(indicator_name))
    if assignments["cluster"].isna().any() or prototypes["cluster"].isna().any():
        raise ValueError("Assignments and prototypes must have cluster labels")
    if prototypes["cluster"].duplicated().any():
        raise ValueError("Prototypes contain duplicate cluster labels")
    if prototypes[feature_names].isna().any().any():
        raise ValueError("Prototypes are missing one or more feature medians")
    replacement_value = config["transformation"]["replace_when"]
    cluster_ids = set(assignments["cluster"])
    prototype_ids = set(prototypes["cluster"])
    if not cluster_ids.issubset(prototype_ids):
        raise ValueError("A predicted cluster has no training prototype")
    prototype_by_cluster = prototypes.set_index("cluster")
    output = test_features[[id_column] + feature_names].copy()
    output["cluster"] = assignments["cluster"].to_numpy()
    output.insert(1, "gdpr", values.to_numpy(dtype=np.int8))
    replacement_rows = output["gdpr"].eq(replacement_value)
    if replacement_rows.any():
        for feature in feature_names:
            output[feature] = output[feature].astype(float)
            output.loc[replacement_rows, feature] = output.loc[
                replacement_rows, "cluster"
            ].map(prototype_by_cluster[feature])
    return {
        "transformed": output[[id_column, "gdpr", "cluster"] + feature_names],
        "original": test_features[[id_column] + feature_names].copy(),
    }


def _require_columns(frame: pd.DataFrame, columns: list, description: str) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("{} must be a DataFrame".format(description))
    if frame.columns.duplicated().any():
        raise ValueError("{} contains duplicate columns".format(description))
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError("{} is missing columns: {}".format(description, missing))
