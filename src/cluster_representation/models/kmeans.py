"""Scikit-learn K-means baseline using the shared pipeline representation."""

from typing import Any, Dict, Mapping

import numpy as np
from sklearn.cluster import KMeans

from .base import ClusterModel


class KMeansModel(ClusterModel):
    def __init__(self, parameters: Mapping[str, Any]) -> None:
        self.parameters = dict(parameters)
        self.estimator = None
        self.labels_ = None

    def fit(self, X: Any) -> "KMeansModel":
        self.estimator = KMeans(**self.parameters).fit(X)
        self.labels_ = np.asarray(self.estimator.labels_, dtype=int) + 1
        return self

    def predict(self, X: Any) -> np.ndarray:
        if self.estimator is None:
            raise RuntimeError("Model has not been fitted")
        return self.estimator.predict(X) + 1

    def metadata(self) -> Dict[str, Any]:
        if self.labels_ is None:
            raise RuntimeError("Model has not been fitted")
        return {
            "name": "kmeans",
            "parameters": self.parameters,
            "constraint_applicability": "not applicable",
            "assignment_strategy": "sklearn KMeans.predict",
            "n_clusters": int(len(np.unique(self.labels_))),
        }
