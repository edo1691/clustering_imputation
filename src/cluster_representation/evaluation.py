"""Clustering and original-unit replacement metrics."""

import json
import math
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    mean_absolute_error,
    silhouette_score,
)

from .randomness import generator


CLUSTER_SELECTION_RULE = (
    "Mean average rank of silhouette (descending), Davies-Bouldin (ascending), "
    "and Calinski-Harabasz (descending); ties select the smaller requested k."
)


def resolve_cluster_selection(
    method: str,
    candidates: List[Mapping[str, Any]],
    configured_selection: Any,
    constraints: Mapping[str, Any],
    constraints_applicable: bool = True,
) -> Dict[str, Any]:
    """Resolve fixed or rank-based cluster selection from training metrics."""
    selection_mode = "auto" if configured_selection == "auto" else "fixed"
    candidate_results = []
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for candidate in candidates:
        requested_k = int(candidate["n_clusters"])
        parameters = _selection_mapping(candidate.get("model_parameters"))
        parameters.pop("n_clusters", None)
        candidate_constraints = _selection_mapping(candidate.get("constraints"))
        if not candidate_constraints and constraints_applicable:
            candidate_constraints = dict(constraints)
        configuration = {
            "parameters": parameters,
            "constraints": candidate_constraints,
        }
        configuration_key = json.dumps(
            configuration,
            sort_keys=True,
            default=str,
        )
        reasons = _selection_exclusion_reasons(
            candidate, constraints, constraints_applicable
        )
        candidate_result = {
            "requested_n_clusters": requested_k,
            "actual_clusters": (
                int(candidate.get("n_clusters"))
                if candidate.get("status") == "success"
                else None
            ),
            "status": candidate.get("status"),
            "failure_reason": candidate.get("failure_reason"),
            "training_time_seconds": candidate.get("training_time_seconds"),
            "configuration": configuration,
            "configuration_key": configuration_key,
            "metrics": {
                "silhouette": _selection_metric(candidate, "silhouette"),
                "davies_bouldin": _selection_metric(candidate, "davies_bouldin"),
                "calinski_harabasz": _selection_metric(candidate, "calinski_harabasz"),
            },
            "metric_ranks": {
                "silhouette": None,
                "davies_bouldin": None,
                "calinski_harabasz": None,
            },
            "selection_score": None,
            "eligible": not reasons,
            "exclusion_reasons": reasons,
        }
        candidate_results.append(candidate_result)
        if not reasons:
            grouped.setdefault(configuration_key, []).append(candidate_result)

    for group in grouped.values():
        rank_values = {
            "silhouette": _average_ranks(
                [item["metrics"]["silhouette"] for item in group], reverse=True
            ),
            "davies_bouldin": _average_ranks(
                [item["metrics"]["davies_bouldin"] for item in group], reverse=False
            ),
            "calinski_harabasz": _average_ranks(
                [item["metrics"]["calinski_harabasz"] for item in group],
                reverse=True,
            ),
        }
        for index, item in enumerate(group):
            item["metric_ranks"] = {
                metric: ranks[index] for metric, ranks in rank_values.items()
            }
            item["selection_score"] = float(
                np.mean(list(item["metric_ranks"].values()))
            )

    if selection_mode == "auto":
        if not grouped:
            details = []
            for candidate in candidate_results:
                details.append(
                    "k={}: {}".format(
                        candidate["requested_n_clusters"],
                        "; ".join(candidate["exclusion_reasons"])
                        or "not eligible",
                    )
                )
            raise ValueError(
                "Automatic cluster selection found no eligible candidates for "
                "{}. {}".format(method, " | ".join(details))
            )
        if len(grouped) != 1:
            raise ValueError(
                "Automatic cluster selection for {} found {} non-equivalent "
                "candidate configuration groups; one run can select only one "
                "configuration.".format(method, len(grouped))
            )
        eligible = next(iter(grouped.values()))
        winner = min(
            eligible,
            key=lambda item: (
                item["selection_score"],
                item["requested_n_clusters"],
            ),
        )
        selected_n_clusters = int(winner["requested_n_clusters"])
    else:
        selected_n_clusters = int(configured_selection)
        if not any(
            item["requested_n_clusters"] == selected_n_clusters
            for item in candidate_results
        ):
            raise ValueError(
                "Fixed cluster selection {} for {} is absent from candidate results".format(
                    selected_n_clusters, method
                )
            )

    return {
        "method": method,
        "selection_mode": selection_mode,
        "configured_selected_n_clusters": configured_selection,
        "selected_n_clusters": selected_n_clusters,
        "selection_rule": CLUSTER_SELECTION_RULE,
        "constraint_applicability": (
            "applicable" if constraints_applicable else "not applicable"
        ),
        "candidate_results": candidate_results,
        "excluded_candidates": [
            {
                "requested_n_clusters": item["requested_n_clusters"],
                "reasons": item["exclusion_reasons"],
            }
            for item in candidate_results
            if not item["eligible"]
        ],
    }


def _selection_mapping(value: Any) -> Dict[str, Any]:
    if isinstance(value, str):
        parsed = json.loads(value)
        return dict(parsed) if isinstance(parsed, Mapping) else {}
    return dict(value) if isinstance(value, Mapping) else {}


def _selection_metric(
    candidate: Mapping[str, Any], metric_name: str
) -> Optional[float]:
    value_keys = {
        "silhouette": "silhouette_score",
        "davies_bouldin": "davies_bouldin_index",
        "calinski_harabasz": "calinski_harabasz_index",
    }
    value = candidate.get(value_keys[metric_name])
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric_value):
        return None
    return numeric_value


def _selection_exclusion_reasons(
    candidate: Mapping[str, Any],
    constraints: Mapping[str, Any],
    constraints_applicable: bool = True,
) -> List[str]:
    reasons = []
    status = candidate.get("status")
    if status != "success":
        reason = "training status is {}".format(status or "unknown")
        if candidate.get("failure_reason"):
            reason += ": {}".format(candidate["failure_reason"])
        reasons.append(reason)

    if constraints_applicable:
        violations = candidate.get("constraint_violations", 0)
        violation_clusters = candidate.get("constraint_violation_clusters", [])
        if violation_clusters is None or (
            isinstance(violation_clusters, (float, np.floating))
            and math.isnan(float(violation_clusters))
        ):
            violation_cluster_count = 0
        else:
            try:
                violation_cluster_count = len(violation_clusters)
            except TypeError:
                violation_cluster_count = int(bool(violation_clusters))
        try:
            numeric_violations = float(violations or 0)
            violation_count = (
                int(numeric_violations) if math.isfinite(numeric_violations) else 0
            )
        except (TypeError, ValueError):
            violation_count = 1
        if violation_count or violation_cluster_count:
            reasons.append(
                "hard constraint violations recorded ({})".format(
                    violation_count or violation_cluster_count
                )
            )
        minimum_size = constraints.get("min_cluster_size")
        actual_minimum = candidate.get("minimum_cluster_size")
        if minimum_size is not None and actual_minimum is not None:
            try:
                actual_minimum_value = float(actual_minimum)
                if not math.isfinite(actual_minimum_value):
                    raise ValueError("non-finite minimum cluster size")
                if int(actual_minimum_value) < int(minimum_size):
                    reasons.append(
                        "minimum cluster size {} is below required {}".format(
                            actual_minimum, minimum_size
                        )
                    )
            except (TypeError, ValueError):
                reasons.append("minimum cluster size is unavailable")

    statuses = {
        "silhouette": candidate.get("silhouette_status"),
        "davies_bouldin": candidate.get("davies_bouldin_status"),
        "calinski_harabasz": candidate.get("calinski_harabasz_status"),
    }
    for metric, status_value in statuses.items():
        if status_value != "available" or _selection_metric(candidate, metric) is None:
            reasons.append(
                "{} metric is unavailable or non-finite ({})".format(
                    metric, status_value or "missing status"
                )
            )
    return reasons


def _average_ranks(values: List[float], reverse: bool) -> List[float]:
    ordered = sorted(set(values), reverse=reverse)
    ranks = {}
    start = 1
    for value in ordered:
        count = values.count(value)
        ranks[value] = (start + start + count - 1) / 2.0
        start += count
    return [ranks[value] for value in values]


def evaluate_training_candidate(
    X: Any, labels: Any, silhouette_indices: np.ndarray
) -> Dict[str, Any]:
    """Evaluate one fitted training partition using a shared silhouette sample."""
    values = np.asarray(X, dtype=float)
    cluster_labels = np.asarray(labels, dtype=int)
    if values.ndim != 2 or cluster_labels.ndim != 1 or len(cluster_labels) != len(values):
        raise ValueError("Training features and fitted labels must align")
    if silhouette_indices.ndim != 1 or (
        len(silhouette_indices)
        and (silhouette_indices.min() < 0 or silhouette_indices.max() >= len(values))
    ):
        raise ValueError("Silhouette sample indices are outside the training data")

    unique, counts = np.unique(cluster_labels, return_counts=True)
    result: Dict[str, Any] = {
        "minimum_cluster_size": int(counts.min()),
        "maximum_cluster_size": int(counts.max()),
        "silhouette_score": None,
        "silhouette_status": "undefined",
        "davies_bouldin_index": None,
        "davies_bouldin_status": "undefined",
        "calinski_harabasz_index": None,
        "calinski_harabasz_status": "undefined",
    }
    if len(unique) < 2 or len(unique) >= len(cluster_labels):
        reason = "requires 2 <= clusters < training rows"
        for key in ("silhouette", "davies_bouldin", "calinski_harabasz"):
            result[key + "_status"] = "undefined: " + reason
        return result

    sample_labels = cluster_labels[silhouette_indices]
    if len(np.unique(sample_labels)) < 2 or len(np.unique(sample_labels)) >= len(sample_labels):
        result["silhouette_status"] = (
            "undefined: the shared silhouette sample must contain 2..n-1 clusters"
        )
    else:
        try:
            result["silhouette_score"] = float(
                silhouette_score(values[silhouette_indices], sample_labels)
            )
            result["silhouette_status"] = "available"
        except ValueError as exc:
            result["silhouette_status"] = "undefined: {}".format(exc)

    for function, value_key, status_key in (
        (
            davies_bouldin_score,
            "davies_bouldin_index",
            "davies_bouldin_status",
        ),
        (
            calinski_harabasz_score,
            "calinski_harabasz_index",
            "calinski_harabasz_status",
        ),
    ):
        try:
            result[value_key] = float(function(values, cluster_labels))
            result[status_key] = "available"
        except ValueError as exc:
            result[status_key] = "undefined: {}".format(exc)
    return result


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
    training_feature_stds: Optional[Mapping[str, float]] = None,
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
        "clustering_metric_feature_space": (
            "shared fitted preprocessing output (median imputation; {} scaling)".format(
                config["preprocessing"]["scaling"]
            )
        ),
        "classification_metrics": {
            "average_precision": {
                "value": None,
                "status": "unavailable_no_ground_truth_labels",
            },
            "f1": {
                "value": None,
                "status": "unavailable_no_ground_truth_labels",
            },
        },
        "scenarios": {},
        "timings_seconds": dict(timings or {}),
        "artifact_sizes_bytes": dict(artifact_sizes or {}),
        "notes": [
            "Overall replacement MAE aggregates different feature scales.",
            "NMAE is the equal-weight mean of per-feature replaced-row MAEs "
            "divided by training standard deviations (ddof=0). Constant "
            "training features are excluded.",
            "Replaced-row MAPE excludes cells with zero original values because "
            "percentage error is undefined for those cells.",
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
    if training_feature_stds is None:
        raise ValueError("Training feature standard deviations are required for NMAE")
    if list(training_feature_stds) != feature_names:
        raise ValueError(
            "Training feature standard deviations must contain exactly the "
            "configured features in order: {}".format(", ".join(feature_names))
        )
    invalid_stds = [
        name
        for name in feature_names
        if not math.isfinite(float(training_feature_stds[name]))
        or float(training_feature_stds[name]) < 0
    ]
    if invalid_stds:
        raise ValueError(
            "Invalid training standard deviation for feature(s): {}".format(
                ", ".join(invalid_stds)
            )
        )
    result["training_feature_scales"] = {
        "feature_stds": {
            name: float(training_feature_stds[name]) for name in feature_names
        },
        "ddof": 0,
        "imputation_policy": config["preprocessing"]["imputation"],
    }
    if test_original[id_column].tolist() != test_assignments[id_column].tolist():
        raise ValueError("Test assignments do not align with original test rows")

    for scenario_name, transformed in transformed_by_scenario.items():
        if transformed[id_column].tolist() != test_original[id_column].tolist():
            raise ValueError("{} rows do not align with original test rows".format(scenario_name))
        if transformed["gdpr"].isna().any() or not transformed["gdpr"].isin([0, 1]).all():
            raise ValueError("{} contains invalid non-consent values".format(scenario_name))
        original_values = test_original[feature_names].to_numpy(dtype=float)
        transformed_values = transformed[feature_names].to_numpy(dtype=float)
        invalid_original = ~np.isfinite(original_values)
        invalid_transformed = ~np.isfinite(transformed_values)
        invalid_features = [
            name
            for index, name in enumerate(feature_names)
            if invalid_original[:, index].any() or invalid_transformed[:, index].any()
        ]
        if invalid_features:
            raise ValueError(
                "Non-finite evaluation values for scenario {!r}, feature(s): {}".format(
                    scenario_name, ", ".join(invalid_features)
                )
            )
        row_errors = np.abs(transformed_values - original_values)
        invalid_error_features = [
            name
            for index, name in enumerate(feature_names)
            if not np.isfinite(row_errors[:, index]).all()
        ]
        if invalid_error_features:
            raise ValueError(
                "Non-finite absolute errors for scenario {!r}, feature(s): {}".format(
                    scenario_name, ", ".join(invalid_error_features)
                )
            )
        replaced_mask = (
            transformed["gdpr"].to_numpy()
            == config["transformation"]["replace_when"]
        )
        per_feature = {
            name: float(mean_absolute_error(
                test_original[name].to_numpy(dtype=float),
                transformed[name].to_numpy(dtype=float),
            ))
            for name in feature_names
        }
        replaced_row_errors = row_errors[replaced_mask]
        replaced_mae = float(replaced_row_errors.mean()) if replaced_mask.any() else None
        replaced_original_values = original_values[replaced_mask]
        replaced_absolute_errors = np.abs(
            transformed_values[replaced_mask] - replaced_original_values
        )
        valid_mape_cells = np.abs(replaced_original_values) != 0
        valid_cell_count = int(valid_mape_cells.sum())
        excluded_zero_cell_count = int(valid_mape_cells.size - valid_cell_count)
        per_feature_replaced_mae: Dict[str, Optional[float]] = {}
        per_feature_replaced_nmae: Dict[str, Optional[float]] = {}
        included_features = []
        excluded_features = {}
        for index, name in enumerate(feature_names):
            feature_std = float(training_feature_stds[name])
            feature_mae = (
                float(replaced_row_errors[:, index].mean())
                if replaced_mask.any()
                else None
            )
            per_feature_replaced_mae[name] = feature_mae
            if feature_std == 0:
                per_feature_replaced_nmae[name] = None
                excluded_features[name] = "zero_training_standard_deviation"
            elif not replaced_mask.any():
                per_feature_replaced_nmae[name] = None
            else:
                normalized_mae = feature_mae / feature_std
                if not math.isfinite(normalized_mae):
                    raise ValueError(
                        "Non-finite normalized MAE for scenario {!r}, feature {!r}".format(
                            scenario_name, name
                        )
                    )
                per_feature_replaced_nmae[name] = normalized_mae
                included_features.append(name)
        if not replaced_mask.any():
            replaced_nmae = None
            replaced_nmae_status = "unavailable_no_replaced_rows"
        elif not included_features:
            replaced_nmae = None
            replaced_nmae_status = "unavailable_no_variable_training_features"
        else:
            replaced_nmae = float(
                np.mean(
                    [
                        per_feature_replaced_nmae[name]
                        for name in included_features
                    ]
                )
            )
            replaced_nmae_status = "available"
        if not replaced_mask.any():
            replaced_mape = None
            replaced_mape_status = "unavailable_no_replaced_rows"
        elif valid_cell_count == 0:
            replaced_mape = None
            replaced_mape_status = "unavailable_no_nonzero_original_values"
        else:
            replaced_mape = float(
                100.0
                * np.mean(
                    replaced_absolute_errors[valid_mape_cells]
                    / np.abs(replaced_original_values[valid_mape_cells])
                )
            )
            replaced_mape_status = "available"
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
            "per_feature_replaced_rows_mae": per_feature_replaced_mae,
            "per_feature_replaced_rows_nmae": per_feature_replaced_nmae,
            "replaced_rows_nmae": replaced_nmae,
            "replaced_rows_nmae_status": replaced_nmae_status,
            "replaced_rows_nmae_included_features": included_features,
            "replaced_rows_nmae_excluded_features": excluded_features,
            "replaced_rows_mape": replaced_mape,
            "replaced_rows_mape_status": replaced_mape_status,
            "replaced_rows_mape_valid_cells": valid_cell_count,
            "replaced_rows_mape_excluded_zero_cells": excluded_zero_cell_count,
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
