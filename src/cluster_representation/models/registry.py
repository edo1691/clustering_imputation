"""Explicit registry for supported and future model templates."""

from typing import Any, Dict

from .base import ClusterModel
from ..robust_support import normalize_robust_config
from .kmeans import KMeansModel
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
    if name in {"robust_coclustering", "robust_constrained_coclustering"}:
        normalized = normalize_robust_config(model_config)
        model = RobustConstrainedCoclustering(normalized["parameters"])
        model.registration_name = name
        if "source_configuration" in normalized:
            model.integration_config_ = normalized["source_configuration"]
        return model
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
    raise ValueError("No model registered as {!r}".format(name))


def training_constraints(model_config: Dict[str, Any]) -> Dict[str, Any]:
    """Translate supported row-size bounds to the pipeline check schema."""
    if model_config["name"] in {"robust_coclustering", "robust_constrained_coclustering"}:
        parameters = algorithm_parameters(model_config)
        return {
            "min_cluster_size": 1 if parameters.get("row_size_min") is None else parameters["row_size_min"],
            "max_cluster_size": parameters.get("row_size_max"),
        }
    return model_config.get("constraints", {})


def algorithm_parameters(model_config: Dict[str, Any]) -> Dict[str, Any]:
    """Map the legacy template configuration to equivalent algorithm arguments."""
    if model_config["name"] in {"robust_coclustering", "robust_constrained_coclustering"}:
        return normalize_robust_config(model_config)["parameters"]
    return dict(model_config["parameters"])
