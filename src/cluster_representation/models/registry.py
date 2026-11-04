"""Explicit registry for supported and future model templates."""

from typing import Any, Dict

from .base import ClusterModel
from .constrained_kmeans import ConstrainedKMeans
from .constrained_coclustering import ConstrainedCoclustering
from .robust_constrained_coclustering import RobustConstrainedCoclustering


def create_model(model_config: Dict[str, Any]) -> ClusterModel:
    name = model_config["name"]
    if name == "constrained_kmeans":
        return ConstrainedKMeans(
            model_config["parameters"], model_config["constraints"]
        )
    if name == "constrained_coclustering":
        return ConstrainedCoclustering(
            model_config["parameters"], model_config["constraints"]
        )
    if name == "robust_constrained_coclustering":
        settings = {
            "constraints": model_config["constraints"],
            "trimming": model_config["trimming"],
        }
        return RobustConstrainedCoclustering(model_config["parameters"], settings)
    raise ValueError("No model registered as {!r}".format(name))
