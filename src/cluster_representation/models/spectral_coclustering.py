"""Unconstrained spectral co-clustering with deterministic row assignment."""

from typing import Any, Dict, Mapping, Optional

import numpy as np
from sklearn.cluster import SpectralCoclustering
from sklearn.metrics import pairwise_distances_argmin

from .base import ClusterModel


class SpectralCoclusteringModel(ClusterModel):
    """Spectral biclustering plus nearest-training-centroid row assignment."""

    assignment_strategy = (
        "nearest_training_centroid; all-zero rows map to the largest training cluster"
    )

    def __init__(self, parameters: Mapping[str, Any]) -> None:
        self.parameters = dict(parameters)
        self.estimator: Optional[SpectralCoclustering] = None
        self.labels_: Optional[np.ndarray] = None
        self.column_labels_: Optional[np.ndarray] = None
        self.feature_min_: Optional[np.ndarray] = None
        self.feature_range_: Optional[np.ndarray] = None
        self.active_columns_: Optional[np.ndarray] = None
        self.centroids_: Optional[np.ndarray] = None
        self.zero_row_cluster_: Optional[int] = None
        self.preprocessing_: Optional[Dict[str, Any]] = None

    def fit(self, X: Any) -> "SpectralCoclusteringModel":
        values = _matrix(X)
        self.feature_min_ = values.min(axis=0)
        maxima = values.max(axis=0)
        self.feature_range_ = maxima - self.feature_min_
        normalized = _nonnegative_transform(
            values, self.feature_min_, self.feature_range_
        )
        self.active_columns_ = np.flatnonzero(np.any(normalized > 0, axis=0))
        if self.active_columns_.size == 0:
            raise ValueError(
                "SpectralCoclustering cannot fit data with no non-zero feature columns"
            )

        model_input = normalized[:, self.active_columns_]
        active_rows = np.flatnonzero(np.any(model_input > 0, axis=1))
        if active_rows.size < 2:
            raise ValueError(
                "SpectralCoclustering requires at least two non-zero training rows"
            )
        n_clusters = self.parameters["n_clusters"]
        if n_clusters > min(active_rows.size, self.active_columns_.size):
            raise ValueError(
                "n_clusters={} exceeds the non-zero training matrix dimensions "
                "({}, {})".format(
                    n_clusters, active_rows.size, self.active_columns_.size
                )
            )

        self.estimator = SpectralCoclustering(
            n_clusters=n_clusters,
            random_state=self.parameters["random_state"],
            n_init=self.parameters["n_init"],
        )
        self.estimator.fit(model_input[active_rows])

        row_labels = np.asarray(self.estimator.row_labels_, dtype=int) + 1
        column_labels = np.full(values.shape[1], -1, dtype=int)
        column_labels[self.active_columns_] = np.asarray(
            self.estimator.column_labels_, dtype=int
        )
        counts = np.bincount(row_labels, minlength=n_clusters + 1)[1:]
        if np.any(counts == 0):
            raise RuntimeError(
                "SpectralCoclustering did not assign training rows to every cluster"
            )
        self.zero_row_cluster_ = int(np.flatnonzero(counts == counts.max())[0] + 1)
        full_labels = np.full(values.shape[0], self.zero_row_cluster_, dtype=int)
        full_labels[active_rows] = row_labels
        self.labels_ = full_labels
        self.column_labels_ = column_labels

        self.centroids_ = np.vstack(
            [
                model_input[self.labels_ == cluster].mean(axis=0)
                for cluster in range(1, n_clusters + 1)
            ]
        )
        self.preprocessing_ = {
            "method": "training_min_max_to_nonnegative",
            "feature_min": self.feature_min_.tolist(),
            "feature_range": self.feature_range_.tolist(),
            "active_column_indices": self.active_columns_.tolist(),
            "inactive_column_indices": np.flatnonzero(
                ~np.isin(np.arange(values.shape[1]), self.active_columns_)
            ).tolist(),
            "below_training_minimum_policy": "clamp normalized values to zero",
            "zero_row_policy": self.assignment_strategy,
        }
        return self

    def predict(self, X: Any) -> np.ndarray:
        if (
            self.feature_min_ is None
            or self.feature_range_ is None
            or self.active_columns_ is None
            or self.centroids_ is None
            or self.labels_ is None
            or self.zero_row_cluster_ is None
        ):
            raise RuntimeError("Model has not been fitted")
        values = _matrix(X)
        if values.shape[1] != len(self.feature_min_):
            raise ValueError("Prediction features do not match the fitted model")
        normalized = _nonnegative_transform(
            values, self.feature_min_, self.feature_range_
        )
        model_input = normalized[:, self.active_columns_]
        zero_rows = ~np.any(model_input > 0, axis=1)
        assignments = np.full(len(values), self.zero_row_cluster_, dtype=int)
        if (~zero_rows).any():
            assignments[~zero_rows] = (
                pairwise_distances_argmin(
                    model_input[~zero_rows], self.centroids_
                )
                + 1
            )
        return assignments

    def metadata(self) -> Dict[str, Any]:
        if self.labels_ is None or self.column_labels_ is None:
            raise RuntimeError("Model has not been fitted")
        return {
            "name": "spectral_coclustering",
            "parameters": self.parameters,
            "constraint_applicability": "not applicable",
            "training_cluster_sizes": {
                str(int(cluster)): int(count)
                for cluster, count in zip(
                    *np.unique(self.labels_, return_counts=True)
                )
            },
            "assignment_strategy": self.assignment_strategy,
            "preprocessing": self.preprocessing_,
            "column_labels": self.column_labels_.tolist(),
        }


def _matrix(X: Any) -> np.ndarray:
    values = np.asarray(X, dtype=float)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("X must be a non-empty two-dimensional matrix")
    if not np.isfinite(values).all():
        raise ValueError("SpectralCoclustering input must contain only finite values")
    return values


def _nonnegative_transform(
    values: np.ndarray, feature_min: np.ndarray, feature_range: np.ndarray
) -> np.ndarray:
    normalized = np.zeros_like(values, dtype=float)
    nonconstant = feature_range > 0
    normalized[:, nonconstant] = (
        values[:, nonconstant] - feature_min[nonconstant]
    ) / feature_range[nonconstant]
    np.maximum(normalized, 0.0, out=normalized)
    if not np.isfinite(normalized).all() or np.any(normalized < 0):
        raise ValueError(
            "SpectralCoclustering preprocessing could not produce finite "
            "nonnegative values"
        )
    return normalized
