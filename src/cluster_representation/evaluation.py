"""Clustering and original-unit replacement metrics."""

from typing import Any, Dict, Mapping, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    mean_absolute_error,
    silhouette_score,
)

from .randomness import generator


def evaluate(
    train_model_inputs: Any,
    train_labels: Any,
    test_original: pd.DataFrame,
    test_assignments: pd.DataFrame,
    transformed_by_scenario: Mapping[str, pd.DataFrame],
    train_sizes: Mapping[int, int],
    requested_rates: Mapping[str, float],
    config: Mapping[str, Any],
    timings: Optional[Mapping[str, float]] = None,
    artifact_sizes: Optional[Mapping[str, int]] = None,
) -> Dict[str, Any]:
    labels = np.asarray(train_labels)
    model_inputs = np.asarray(train_model_inputs, dtype=float)
    result: Dict[str, Any] = {
        "train_size": int(len(labels)),
        "test_size": int(len(test_original)),
        "n_clusters": int(len(np.unique(labels))),
        "training_cluster_sizes": {
            str(int(cluster)): int(size) for cluster, size in train_sizes.items()
        },
        "minimum_training_cluster_size": int(min(train_sizes.values()))
        if train_sizes
        else None,
        "average_training_cluster_size": float(np.mean(list(train_sizes.values())))
        if train_sizes
        else None,
        "constraint_violations": [],
        "clustering_metrics": _clustering_metrics(
            model_inputs, labels, config
        ),
        "scenarios": {},
        "timings_seconds": dict(timings or {}),
        "artifact_sizes_bytes": dict(artifact_sizes or {}),
        "notes": [
            "Overall replacement MAE aggregates different feature scales.",
            "Original transactional features are accessible for cluster assignment. "
            "For simulated non-consenting customers, their direct downstream "
            "representation is replaced with training-derived cluster medians.",
            "This transformation alone does not establish anonymity or regulatory compliance.",
        ],
    }
    required_minimum = config.model_config.get("constraints", {}).get(
        "min_cluster_size"
    )
    if required_minimum is not None:
        result["constraint_violations"] = [
            {"cluster": int(cluster), "size": int(size)}
            for cluster, size in train_sizes.items()
            if size < required_minimum
        ]
    id_column = config["dataset"]["id_column"]
    feature_names = [feature["name"] for feature in config["features"]]
    if test_original[id_column].tolist() != test_assignments[id_column].tolist():
        raise ValueError("Test assignments do not align with original test rows")

    for scenario_name, transformed in transformed_by_scenario.items():
        if transformed[id_column].tolist() != test_original[id_column].tolist():
            raise ValueError("{} rows do not align with original test rows".format(scenario_name))
        if transformed["gdpr"].isna().any() or not transformed["gdpr"].isin([0, 1]).all():
            raise ValueError("{} contains invalid non-consent values".format(scenario_name))
        expected_replaced = config["transformation"]["replace_when"]
        row_errors = np.abs(
            transformed[feature_names].to_numpy(dtype=float)
            - test_original[feature_names].to_numpy(dtype=float)
        )
        per_feature = {
            name: float(mean_absolute_error(
                test_original[name].to_numpy(dtype=float),
                transformed[name].to_numpy(dtype=float),
            ))
            for name in feature_names
        }
        replaced_mask = transformed["gdpr"].to_numpy() == expected_replaced
        replaced_mae = (
            float(row_errors[replaced_mask].mean())
            if replaced_mask.any()
            else None
        )
        requested = requested_rates.get(scenario_name, {})
        result["scenarios"][scenario_name] = {
            "requested_nonconsent_rate": requested.get("requested"),
            "realized_overall_nonconsent_rate": requested.get("overall"),
            "realized_train_nonconsent_rate": requested.get("train"),
            "realized_test_nonconsent_rate": float(transformed["gdpr"].mean()),
            "n_replaced": int(replaced_mask.sum()),
            "per_feature_mae": per_feature,
            "overall_test_mae": float(row_errors.mean()),
            "replaced_rows_mae": replaced_mae,
            "replaced_rows_mae_status": (
                "available" if replaced_mask.any() else "unavailable_no_replaced_rows"
            ),
        }
    return result


def _clustering_metrics(
    X: np.ndarray, labels: np.ndarray, config: Mapping[str, Any]
) -> Dict[str, Any]:
    distinct = np.unique(labels)
    n_samples = len(labels)
    if len(distinct) < 2 or len(distinct) >= n_samples:
        unavailable = {
            "value": None,
            "status": "undefined: requires 2 <= clusters < samples",
        }
        return {
            "silhouette": dict(unavailable),
            "davies_bouldin": dict(unavailable),
            "calinski_harabasz": dict(unavailable),
        }
    sample_limit = config["evaluation"]["silhouette_sample_size"]
    if n_samples > sample_limit:
        rng = generator(config["dataset"]["seed"], "evaluation:silhouette")
        selected = np.sort(rng.choice(n_samples, size=sample_limit, replace=False))
        X_sample, labels_sample = X[selected], labels[selected]
    else:
        X_sample, labels_sample = X, labels
    metrics = {}
    for key, function, data, target in (
        ("silhouette", silhouette_score, X_sample, labels_sample),
        ("davies_bouldin", davies_bouldin_score, X, labels),
        ("calinski_harabasz", calinski_harabasz_score, X, labels),
    ):
        try:
            metrics[key] = {"value": float(function(data, target)), "status": "available"}
        except ValueError as exc:
            metrics[key] = {"value": None, "status": "undefined: {}".format(exc)}
    return metrics
