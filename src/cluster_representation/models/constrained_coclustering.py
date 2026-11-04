"""Unimplemented template for future constrained co-clustering."""

from typing import Any, Dict, Mapping

from .base import ClusterModel


class ConstrainedCoclustering(ClusterModel):
    """Configuration placeholder; no co-clustering behavior is simulated."""

    def __init__(self, parameters: Mapping[str, Any], constraints: Mapping[str, Any]) -> None:
        self.parameters = dict(parameters)
        self.constraints = dict(constraints)
        self.labels_ = None
        self.row_memberships_ = None
        self.column_memberships_ = None
        self.fitted_parameters_ = None
        self.prototype_support_ = None

    def fit(self, X: Any) -> "ConstrainedCoclustering":
        raise NotImplementedError(
            "constrained_coclustering is a configuration template only; "
            "algorithm design is future work"
        )

    def predict(self, X: Any) -> Any:
        raise NotImplementedError(
            "constrained_coclustering prediction is not implemented"
        )

    def metadata(self) -> Dict[str, Any]:
        return {
            "name": "constrained_coclustering",
            "parameters": self.parameters,
            "constraints": self.constraints,
            "implemented": False,
        }
