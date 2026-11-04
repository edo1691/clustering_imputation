"""Minimum-size-constrained K-means using k-means-constrained."""

from typing import Any, Dict, Mapping, Optional

import numpy as np
from k_means_constrained import KMeansConstrained
from sklearn.metrics import pairwise_distances_argmin

from .base import ClusterModel


class ConstrainedKMeans(ClusterModel):
    def __init__(self, parameters: Mapping[str, Any], constraints: Mapping[str, Any]) -> None:
        self.parameters = dict(parameters)
        self.constraints = dict(constraints)
        self.estimator: Optional[KMeansConstrained] = None
        self.labels_: Optional[np.ndarray] = None

    def fit(self, X: Any) -> "ConstrainedKMeans":
        values = np.asarray(X, dtype=float)
        if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
            raise ValueError("X must be a non-empty two-dimensional matrix")
        if not np.isfinite(values).all():
            raise ValueError("X must contain only finite values")
        n_clusters = self.parameters["n_clusters"]
        min_size = self.constraints["min_cluster_size"]
        if n_clusters * min_size > values.shape[0]:
            raise ValueError(
                "Infeasible minimum cluster size: n_clusters * min_cluster_size "
                "exceeds the number of training rows"
            )
        self.estimator = KMeansConstrained(
            n_clusters=n_clusters,
            size_min=min_size,
            random_state=self.parameters["random_state"],
            max_iter=self.parameters["max_iter"],
            n_init=self.parameters["n_init"],
        )
        self.estimator.fit(values)
        internal = np.asarray(self.estimator.labels_, dtype=int)
        counts = np.bincount(internal, minlength=n_clusters)
        if len(counts) != n_clusters or np.any(counts < min_size):
            raise RuntimeError(
                "Constrained K-means returned cluster sizes below the configured minimum"
            )
        self.labels_ = internal + 1
        return self

    def predict(self, X: Any) -> np.ndarray:
        if self.estimator is None:
            raise RuntimeError("Model has not been fitted")
        values = np.asarray(X, dtype=float)
        if values.ndim != 2 or values.shape[1] != self.estimator.cluster_centers_.shape[1]:
            raise ValueError("Prediction features do not match the fitted model")
        if not np.isfinite(values).all():
            raise ValueError("Prediction features must contain only finite values")
        return pairwise_distances_argmin(values, self.estimator.cluster_centers_) + 1

    @property
    def cluster_centers_(self) -> np.ndarray:
        if self.estimator is None:
            raise RuntimeError("Model has not been fitted")
        return self.estimator.cluster_centers_

    @property
    def cluster_label_mapping(self) -> Dict[int, int]:
        return {index: index + 1 for index in range(self.parameters["n_clusters"])}

    def metadata(self) -> Dict[str, Any]:
        return {
            "name": "constrained_kmeans",
            "parameters": self.parameters,
            "constraints": self.constraints,
            "cluster_label_mapping": self.cluster_label_mapping,
        }
