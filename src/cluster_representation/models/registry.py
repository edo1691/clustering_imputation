"""Explicit registry for supported and future model templates."""

from typing import Any, Dict

from .base import ClusterModel
from .kmeans import KMeansModel
from .robust_coclustering import RobustCoclustering
from .constrained_kmeans import ConstrainedKMeans
from .constrained_coclustering import ConstrainedCoclustering
from .robust_constrained_coclustering import RobustConstrainedCoclustering
from .spectral_coclustering import SpectralCoclusteringModel


MODEL_NAMES = (
    "kmeans",
    "robust_coclustering",
    "constrained_kmeans",
    "constrained_coclustering",
    "robust_constrained_coclustering",
    "spectral_coclustering",
)


def create_model(model_config: Dict[str, Any]) -> ClusterModel:
    name = model_config["name"]
    if name == "kmeans":
        return KMeansModel(model_config["parameters"])
    if name == "robust_coclustering":
        return RobustCoclustering(model_config["parameters"])
    if name == "constrained_kmeans":
        return ConstrainedKMeans(
            model_config["parameters"], model_config["constraints"]
        )
    if name == "spectral_coclustering":
        return SpectralCoclusteringModel(model_config["parameters"])
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


def training_constraints(model_config: Dict[str, Any]) -> Dict[str, Any]:
    """Translate supported row-size bounds to the pipeline check schema."""
    if model_config["name"] == "robust_coclustering":
        parameters = model_config["parameters"]
        return {
            "min_cluster_size": 1 if parameters.get("row_size_min") is None else parameters["row_size_min"],
            "max_cluster_size": parameters.get("row_size_max"),
        }
    return model_config.get("constraints", {})
