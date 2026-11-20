"""Pipeline adapter for the unmodified trimmed Gaussian CEM implementation."""

from typing import Any, Dict, Mapping

import numpy as np

from .base import ClusterModel
from .custom import robcoclust_constrained as algorithm


class RobustConstrainedCoclustering(ClusterModel):
    """Keep algorithm outputs intact and expose one-based pipeline labels."""

    assignment_strategy = "fixed Gaussian posterior with iterative batch cell trimming"

    def __init__(self, parameters: Mapping[str, Any]) -> None:
        self.parameters = dict(parameters)
        self.result_ = None
        self.labels_ = None
        self.column_labels_ = None
        self.preprocessing_ = {
            "method": "shared training-fitted imputation and scaling",
            "model_input": "rows x features, univariate Gaussian cells (d=1)",
            "feature_filtering": "none",
        }

    def fit(self, X: Any) -> "RobustConstrainedCoclustering":
        values = _matrix(X)
        result = algorithm.tbcem_constrained(values, **self.parameters)
        row_labels = _hard_labels(result["z"], values.shape[0], self.parameters["g"])
        column_labels = _hard_labels(result["w"], values.shape[1], self.parameters["m"])
        for labels, minimum, maximum, count in (
            (row_labels, self.parameters.get("row_size_min"), self.parameters.get("row_size_max"), self.parameters["g"]),
            (column_labels, self.parameters.get("col_size_min"), self.parameters.get("col_size_max"), self.parameters["m"]),
        ):
            sizes = np.bincount(labels, minlength=count + 1)[1:]
            if np.any(sizes < (1 if minimum is None else minimum)) or (
                maximum is not None and np.any(sizes > maximum)
            ):
                raise RuntimeError("Returned memberships violate configured size bounds")
        self.result_ = result
        self.labels_ = row_labels
        self.column_labels_ = column_labels
        self.n_features_in_ = values.shape[1]
        return self

    def predict(self, X: Any) -> np.ndarray:
        if self.result_ is None:
            raise RuntimeError("Model has not been fitted")
        values = _matrix(X)
        if values.shape[1] != self.n_features_in_:
            raise ValueError("Prediction features do not match the fitted model")
        result = self.result_
        mask0 = (~np.isnan(values)).astype(int)
        x = np.where(mask0 == 1, values, 0.0)[..., None]
        gamma = (result["mu"], result["sigma"])
        posterior = algorithm._row_posterior_mn(x, result["w"], result["pi"], gamma, mask0)
        labels = posterior.argmax(axis=1)
        previous_mask = None
        for iteration in range(self.parameters.get("iter_max", 50)):
            z = np.eye(self.parameters["g"], dtype=int)[labels]
            mask = algorithm._impartial_trimming(
                x, z, result["w"], *gamma, mask0,
                self.parameters.get("trim_lvl", 0.10),
            )
            posterior = algorithm._row_posterior_mn(x, result["w"], result["pi"], gamma, mask)
            next_labels = posterior.argmax(axis=1)
            if np.array_equal(labels, next_labels) and np.array_equal(previous_mask, mask):
                break
            labels = next_labels
            previous_mask = mask.copy()
        else:
            raise RuntimeError("Robust prediction did not reach stable labels and mask within iter_max")
        self.prediction_diagnostics_ = {
            "iterations": iteration + 1,
            "trimmed_cells": int((mask0 - mask).sum()),
            "all_masked_rows": int((mask.sum(axis=1) == 0).sum()),
        }
        return next_labels + 1

    def metadata(self) -> Dict[str, Any]:
        if self.result_ is None:
            raise RuntimeError("Model has not been fitted")
        result = self.result_
        return {
            "name": getattr(self, "registration_name", "robust_coclustering"),
            "integration_config": getattr(self, "integration_config_", None),
            "parameters": self.parameters,
            "n_clusters": int(len(np.unique(self.labels_))),
            "n_column_clusters": int(len(np.unique(self.column_labels_))),
            "constraints": {
                "min_cluster_size": 1 if self.parameters.get("row_size_min") is None else self.parameters["row_size_min"],
                "max_cluster_size": self.parameters.get("row_size_max"),
                "col_size_min": self.parameters.get("col_size_min"),
                "col_size_max": self.parameters.get("col_size_max"),
            },
            "constraint_applicability": "applicable",
            "assignment_strategy": self.assignment_strategy,
            "prediction_size_constraints": "not applicable",
            "preprocessing": self.preprocessing_,
            "column_labels": self.column_labels_.tolist(),
            "column_cluster_sizes": {
                str(int(label)): int(count)
                for label, count in zip(*np.unique(self.column_labels_, return_counts=True))
            },
            "column_constraint_violations": [],
            "training_cluster_sizes": {
                str(int(label)): int(count)
                for label, count in zip(*np.unique(self.labels_, return_counts=True))
            },
            "score": float(result["score"]),
            "converged": bool(result["converged"]),
            "n_iter": int(result["n_iter"]),
            "n_success": int(result["n_success"]),
            "n_fail": int(result["n_fail"]),
            "best_init": int(result["best_init"]),
            "algorithm_outputs": "complete result_ dictionary in model.joblib",
        }


def _matrix(X: Any) -> np.ndarray:
    values = np.asarray(X, dtype=float)
    if values.ndim != 2 or 0 in values.shape:
        raise ValueError("X must be a non-empty two-dimensional matrix")
    if np.isinf(values).any():
        raise ValueError("Robust input must not contain infinities")
    return values


def _hard_labels(membership: Any, rows: int, groups: int) -> np.ndarray:
    values = np.asarray(membership)
    if (values.shape != (rows, groups)
            or not np.isin(values, [0, 1]).all()
            or not (values.sum(axis=1) == 1).all()):
        raise ValueError("Expected hard one-hot memberships; conversion requires explicit approval")
    return values.argmax(axis=1) + 1
