"""Strict YAML configuration loading and validation."""

import copy
import hashlib
import json
import math
import re
from math import ceil
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Set

import yaml

from .models.registry import MODEL_NAMES


PIPELINE_STEPS = (
    "generate_dataset",
    "split_dataset",
    "fit_preprocessor",
    "train_model",
    "build_prototypes",
    "transform_test",
    "evaluate",
)

_DEPENDENCIES = {
    "generate_dataset": set(),
    "split_dataset": {"generate_dataset"},
    "fit_preprocessor": {"split_dataset"},
    "train_model": {"fit_preprocessor"},
    "build_prototypes": {"train_model"},
    "transform_test": {"build_prototypes"},
    "evaluate": {"transform_test"},
}
_DISTRIBUTION_PARAMETERS = {
    "normal": {"mean", "std"},
    "uniform": {"low", "high"},
    "exponential": {"scale"},
    "poisson": {"lam"},
    "binomial": {"n", "p"},
    "lognormal": {"mean", "sigma"},
}


class ConfigurationError(ValueError):
    """Raised when a configuration is invalid."""


class LoadedConfig(dict):
    """Validated settings with resolved source and model configuration paths."""

    def __init__(
        self,
        settings: Dict[str, Any],
        source_path: Path,
        model_config: Dict[str, Any],
        model_config_path: Path,
        models_run: Optional[list] = None,
        model_configs: Optional[Dict[str, Dict[str, Any]]] = None,
        model_config_paths: Optional[Dict[str, Path]] = None,
    ) -> None:
        super().__init__(settings)
        self.source_path = source_path
        self.model_config = model_config
        self.model_config_path = model_config_path
        self.models_run = list(models_run or [model_config["name"]])
        self.model_configs = dict(model_configs or {model_config["name"]: model_config})
        self.model_config_paths = dict(
            model_config_paths or {model_config["name"]: model_config_path}
        )

    def for_model(self, name: str) -> "LoadedConfig":
        if name not in self.models_run:
            raise KeyError("Model {!r} is not selected in models.run".format(name))
        return LoadedConfig(
            copy.deepcopy(dict(self)),
            self.source_path,
            copy.deepcopy(self.model_configs[name]),
            self.model_config_paths[name],
            self.models_run,
            self.model_configs,
            self.model_config_paths,
        )

    @property
    def artifact_dir(self) -> Path:
        return Path(self["output"]["root"]) / self["output"]["run_name"]

    @property
    def fingerprint(self) -> str:
        serializable = copy.deepcopy(dict(self))
        serializable["models"]["configs"] = {
            name: str(path) for name, path in self.model_config_paths.items()
        }
        serializable["resolved_model_configs"] = self.model_configs
        payload = json.dumps(serializable, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _mapping(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ConfigurationError("{} must be a mapping".format(where))
    return value


def _keys(mapping: Mapping[str, Any], allowed: Set[str], where: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise ConfigurationError(
            "{} contains unknown key(s): {}".format(where, ", ".join(unknown))
        )


def _require(
    mapping: Mapping[str, Any], required: Iterable[str], where: str
) -> None:
    missing = sorted(set(required) - set(mapping))
    if missing:
        raise ConfigurationError(
            "{} is missing required key(s): {}".format(where, ", ".join(missing))
        )


def _integer(value: Any, where: str, minimum: Optional[int] = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigurationError("{} must be an integer".format(where))
    if minimum is not None and value < minimum:
        raise ConfigurationError("{} must be at least {}".format(where, minimum))
    return value


def _number(value: Any, where: str, minimum: Optional[float] = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError("{} must be numeric".format(where))
    number = float(value)
    if not math.isfinite(number):
        raise ConfigurationError("{} must be finite".format(where))
    if minimum is not None and number < minimum:
        raise ConfigurationError("{} must be at least {}".format(where, minimum))
    return number


def _validate_feature_list(features: Any) -> None:
    if not isinstance(features, list) or not features:
        raise ConfigurationError("features must be a non-empty list")
    expected_names = ["ft_{}".format(index) for index in range(1, len(features) + 1)]
    seen = []
    for index, raw_feature in enumerate(features):
        where = "features[{}]".format(index)
        feature = _mapping(raw_feature, where)
        _keys(feature, {"name", "distribution", "parameters"}, where)
        _require(feature, {"name", "distribution", "parameters"}, where)
        name = feature["name"]
        if not isinstance(name, str) or not re.fullmatch(r"ft_[1-9][0-9]*", name):
            raise ConfigurationError(
                "{}.name must follow ft_1, ft_2, ...".format(where)
            )
        seen.append(name)
        distribution = feature["distribution"]
        if not isinstance(distribution, str) or distribution not in _DISTRIBUTION_PARAMETERS:
            raise ConfigurationError(
                "{} has unsupported distribution {!r}".format(
                    where, distribution
                )
            )
        parameters = _mapping(feature["parameters"], "{}.parameters".format(where))
        expected = _DISTRIBUTION_PARAMETERS[distribution]
        _keys(parameters, expected, "{}.parameters".format(where))
        _require(parameters, expected, "{}.parameters".format(where))
        if distribution == "normal":
            _number(parameters["mean"], where + ".parameters.mean")
            _number(parameters["std"], where + ".parameters.std", 0)
            if parameters["std"] == 0:
                raise ConfigurationError(where + ".parameters.std must be positive")
        elif distribution == "uniform":
            low = _number(parameters["low"], where + ".parameters.low")
            high = _number(parameters["high"], where + ".parameters.high")
            if low >= high:
                raise ConfigurationError(where + ".parameters.low must be below high")
        elif distribution in ("exponential", "lognormal"):
            parameter = "scale" if distribution == "exponential" else "sigma"
            if distribution == "lognormal":
                _number(parameters["mean"], where + ".parameters.mean")
            size = _number(parameters[parameter], where + ".parameters." + parameter, 0)
            if size == 0:
                raise ConfigurationError(
                    "{}.parameters.{} must be positive".format(where, parameter)
                )
        elif distribution == "poisson":
            _number(parameters["lam"], where + ".parameters.lam", 0)
        else:
            _integer(parameters["n"], where + ".parameters.n", 0)
            probability = _number(parameters["p"], where + ".parameters.p")
            if not 0 <= probability <= 1:
                raise ConfigurationError(where + ".parameters.p must be in [0, 1]")
    if len(set(seen)) != len(seen):
        raise ConfigurationError("feature names must be unique")
    if seen != expected_names:
        raise ConfigurationError(
            "feature names must be contiguous and ordered: {}".format(
                ", ".join(expected_names)
            )
        )


def _validate_model_config(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            model = yaml.safe_load(stream)
    except OSError as exc:
        raise ConfigurationError("Cannot read model config {}: {}".format(path, exc))
    _mapping(model, "model configuration")
    name = model.get("name")
    if name == "spectral_coclustering":
        _keys(
            model,
            {"name", "parameters", "cluster_search"},
            "model configuration",
        )
        _require(model, {"name", "parameters", "cluster_search"}, "model configuration")
        parameters = _mapping(model["parameters"], "model.parameters")
        _keys(parameters, {"n_clusters", "random_state", "n_init"}, "model.parameters")
        _require(parameters, {"random_state", "n_init"}, "model.parameters")
        _integer(parameters["random_state"], "model.parameters.random_state", 0)
        _integer(parameters["n_init"], "model.parameters.n_init", 1)

        search = _mapping(model["cluster_search"], "model.cluster_search")
        _keys(
            search,
            {"enabled", "candidates", "selected_n_clusters"},
            "model.cluster_search",
        )
        _require(
            search,
            {"enabled", "candidates", "selected_n_clusters"},
            "model.cluster_search",
        )
        if not isinstance(search["enabled"], bool):
            raise ConfigurationError("model.cluster_search.enabled must be boolean")
        candidates = search["candidates"]
        if not isinstance(candidates, list) or not candidates:
            raise ConfigurationError(
                "model.cluster_search.candidates must be a non-empty list"
            )
        for index, candidate in enumerate(candidates):
            _integer(
                candidate,
                "model.cluster_search.candidates[{}]".format(index),
                2,
            )
        if len(set(candidates)) != len(candidates):
            raise ConfigurationError(
                "model.cluster_search.candidates must contain unique values"
            )
        configured_selection = search["selected_n_clusters"]
        if configured_selection == "auto":
            if not search["enabled"]:
                raise ConfigurationError(
                    "model.cluster_search.selected_n_clusters='auto' requires "
                    "cluster_search.enabled=true"
                )
            selected = "auto"
        else:
            selected = _integer(
                configured_selection,
                "model.cluster_search.selected_n_clusters",
                2,
            )
            if selected not in candidates:
                raise ConfigurationError(
                    "model.cluster_search.selected_n_clusters must be a candidate"
                )
        if "n_clusters" in parameters:
            configured_n_clusters = _integer(
                parameters["n_clusters"], "model.parameters.n_clusters", 2
            )
            if search["enabled"]:
                raise ConfigurationError(
                    "model.parameters.n_clusters must be omitted when cluster search is enabled"
                )
            if configured_n_clusters != selected:
                raise ConfigurationError(
                    "model.parameters.n_clusters must match selected_n_clusters "
                    "when cluster search is disabled"
                )
        if not search["enabled"] and "n_clusters" not in parameters:
            parameters["n_clusters"] = selected
        return model
    if name == "constrained_kmeans":
        allowed_model_keys = {"name", "parameters", "constraints", "cluster_search"}
        _keys(model, allowed_model_keys, "model configuration")
        _require(model, {"name", "parameters", "constraints"}, "model configuration")
        parameters = _mapping(model["parameters"], "model.parameters")
        search = model.get("cluster_search")
        if search is None:
            _keys(
                parameters,
                {"n_clusters", "random_state", "max_iter", "n_init"},
                "model.parameters",
            )
            _require(
                parameters,
                {"n_clusters", "random_state", "max_iter", "n_init"},
                "model.parameters",
            )
            _integer(parameters["n_clusters"], "model.parameters.n_clusters", 2)
        else:
            _keys(
                parameters,
                {"n_clusters", "random_state", "max_iter", "n_init"},
                "model.parameters",
            )
            _require(parameters, {"random_state", "max_iter", "n_init"}, "model.parameters")
            search = _mapping(search, "model.cluster_search")
            _keys(
                search,
                {"enabled", "candidates", "selected_n_clusters"},
                "model.cluster_search",
            )
            _require(
                search,
                {"enabled", "candidates", "selected_n_clusters"},
                "model.cluster_search",
            )
            if not isinstance(search["enabled"], bool):
                raise ConfigurationError("model.cluster_search.enabled must be boolean")
            candidates = search["candidates"]
            if not isinstance(candidates, list) or not candidates:
                raise ConfigurationError(
                    "model.cluster_search.candidates must be a non-empty list"
                )
            for index, candidate in enumerate(candidates):
                _integer(
                    candidate,
                    "model.cluster_search.candidates[{}]".format(index),
                    1,
                )
            if len(set(candidates)) != len(candidates):
                raise ConfigurationError(
                    "model.cluster_search.candidates must contain unique values"
                )
            configured_selection = search["selected_n_clusters"]
            if configured_selection == "auto":
                if not search["enabled"]:
                    raise ConfigurationError(
                        "model.cluster_search.selected_n_clusters='auto' requires "
                        "cluster_search.enabled=true"
                    )
                selected = "auto"
            else:
                selected = _integer(
                    configured_selection,
                    "model.cluster_search.selected_n_clusters",
                    1,
                )
                if selected not in candidates:
                    raise ConfigurationError(
                        "model.cluster_search.selected_n_clusters must be a candidate"
                    )
            if "n_clusters" in parameters:
                configured_n_clusters = _integer(
                    parameters["n_clusters"], "model.parameters.n_clusters", 1
                )
                if search["enabled"]:
                    raise ConfigurationError(
                        "model.parameters.n_clusters must be omitted when cluster search is enabled"
                    )
                if selected == "auto":
                    raise ConfigurationError(
                        "model.parameters.n_clusters must be omitted when "
                        "selected_n_clusters='auto'"
                    )
                if configured_n_clusters != selected:
                    raise ConfigurationError(
                        "model.parameters.n_clusters must match selected_n_clusters "
                        "when cluster search is disabled"
                    )
            if not search["enabled"] and "n_clusters" not in parameters:
                parameters["n_clusters"] = selected
        _integer(parameters["random_state"], "model.parameters.random_state", 0)
        _integer(parameters["max_iter"], "model.parameters.max_iter", 1)
        _integer(parameters["n_init"], "model.parameters.n_init", 1)
        constraints = _mapping(model["constraints"], "model.constraints")
        _keys(constraints, {"type", "min_cluster_size"}, "model.constraints")
        _require(constraints, {"type", "min_cluster_size"}, "model.constraints")
        if constraints["type"] != "minimum_cluster_size":
            raise ConfigurationError(
                "model.constraints.type must be 'minimum_cluster_size'"
            )
        _integer(
            constraints["min_cluster_size"],
            "model.constraints.min_cluster_size",
            1,
        )
    elif name in {"constrained_coclustering", "robust_constrained_coclustering"}:
        allowed_model_keys = {"name", "parameters", "constraints"}
        required_model_keys = {"name", "parameters", "constraints"}
        if name == "robust_constrained_coclustering":
            allowed_model_keys.add("trimming")
            required_model_keys.add("trimming")
        _keys(model, allowed_model_keys, "model configuration")
        _require(model, required_model_keys, "model configuration")
        parameters = _mapping(model["parameters"], "model.parameters")
        _keys(
            parameters,
            {"n_row_clusters", "n_column_clusters", "random_state", "max_iter", "n_init"},
            "model.parameters",
        )
        _require(
            parameters,
            {"n_row_clusters", "n_column_clusters", "random_state", "max_iter", "n_init"},
            "model.parameters",
        )
        _integer(parameters["n_row_clusters"], "model.parameters.n_row_clusters", 2)
        _integer(parameters["n_column_clusters"], "model.parameters.n_column_clusters", 1)
        _integer(parameters["random_state"], "model.parameters.random_state", 0)
        _integer(parameters["max_iter"], "model.parameters.max_iter", 1)
        _integer(parameters["n_init"], "model.parameters.n_init", 1)
        constraints = _mapping(model["constraints"], "model.constraints")
        _keys(constraints, {"min_row_cluster_size"}, "model.constraints")
        _require(constraints, {"min_row_cluster_size"}, "model.constraints")
        _integer(constraints["min_row_cluster_size"], "min_row_cluster_size", 1)
        if name == "robust_constrained_coclustering":
            trimming = _mapping(model["trimming"], "model.trimming")
            _keys(trimming, {"row_fraction", "column_fraction"}, "model.trimming")
            _require(trimming, {"row_fraction", "column_fraction"}, "model.trimming")
            for key in ("row_fraction", "column_fraction"):
                fraction = _number(trimming[key], "model.trimming." + key)
                if not 0 <= fraction < 1:
                    raise ConfigurationError(
                        "model.trimming.{} must be in [0, 1)".format(key)
                    )
    else:
        raise ConfigurationError("Unsupported model name: {!r}".format(name))
    return model


def _validate(settings: Any, source_path: Path) -> LoadedConfig:
    root = _mapping(settings, "configuration")
    top_keys = {
        "dataset",
        "features",
        "nonconsent",
        "split",
        "preprocessing",
        "models",
        "prototypes",
        "transformation",
        "evaluation",
        "pipeline",
        "output",
    }
    _keys(root, top_keys, "configuration")
    _require(root, top_keys, "configuration")

    dataset = _mapping(root["dataset"], "dataset")
    _keys(dataset, {"n_rows", "seed", "id_column"}, "dataset")
    _require(dataset, {"n_rows", "seed", "id_column"}, "dataset")
    n_rows = _integer(dataset["n_rows"], "dataset.n_rows", 2)
    _integer(dataset["seed"], "dataset.seed", 0)
    if not isinstance(dataset["id_column"], str) or not dataset["id_column"]:
        raise ConfigurationError("dataset.id_column must be a non-empty string")
    _validate_feature_list(root["features"])

    nonconsent = _mapping(root["nonconsent"], "nonconsent")
    _keys(
        nonconsent,
        {
            "prefix",
            "start_percent",
            "end_percent",
            "step_percent",
            "assignment",
            "count_rounding",
        },
        "nonconsent",
    )
    _require(
        nonconsent,
        {
            "prefix",
            "start_percent",
            "end_percent",
            "step_percent",
            "assignment",
            "count_rounding",
        },
        "nonconsent",
    )
    if not isinstance(nonconsent["prefix"], str) or not nonconsent["prefix"]:
        raise ConfigurationError("nonconsent.prefix must be a non-empty string")
    start = _integer(nonconsent["start_percent"], "nonconsent.start_percent", 0)
    end = _integer(nonconsent["end_percent"], "nonconsent.end_percent", 0)
    step = _integer(nonconsent["step_percent"], "nonconsent.step_percent", 1)
    if start > 100 or end > 100 or start > end:
        raise ConfigurationError("non-consent percentages must satisfy 0 <= start <= end <= 100")
    if (end - start) % step:
        raise ConfigurationError("nonconsent.step_percent must reach end_percent exactly")
    if not isinstance(nonconsent["assignment"], str) or nonconsent["assignment"] not in {
        "independent",
        "nested",
    }:
        raise ConfigurationError("nonconsent.assignment must be independent or nested")
    if nonconsent["count_rounding"] != "floor":
        raise ConfigurationError("nonconsent.count_rounding only supports 'floor'")

    split = _mapping(root["split"], "split")
    _keys(split, {"test_size", "seed"}, "split")
    _require(split, {"test_size", "seed"}, "split")
    test_size = _number(split["test_size"], "split.test_size")
    n_test = int(ceil(n_rows * test_size))
    if not 0 < test_size < 1 or n_test in (0, n_rows):
        raise ConfigurationError("split.test_size must produce non-empty train and test splits")
    _integer(split["seed"], "split.seed", 0)

    preprocessing = _mapping(root["preprocessing"], "preprocessing")
    _keys(preprocessing, {"imputation", "scaling"}, "preprocessing")
    _require(preprocessing, {"imputation", "scaling"}, "preprocessing")
    if preprocessing["imputation"] != "median":
        raise ConfigurationError("preprocessing.imputation only supports 'median'")
    if not isinstance(preprocessing["scaling"], str) or preprocessing["scaling"] not in {
        "standard",
        "none",
    }:
        raise ConfigurationError("preprocessing.scaling must be 'standard' or 'none'")

    models = _mapping(root["models"], "models")
    _keys(models, {"run", "configs"}, "models")
    _require(models, {"run", "configs"}, "models")
    models_run = models["run"]
    if not isinstance(models_run, list) or not models_run:
        raise ConfigurationError("models.run must contain at least one model name")
    if any(not isinstance(name, str) for name in models_run):
        raise ConfigurationError("models.run entries must be model names")
    unknown_models = sorted(set(models_run) - set(MODEL_NAMES))
    if unknown_models:
        raise ConfigurationError(
            "Unknown model name(s) in models.run: {}. Valid model names: {}".format(
                ", ".join(unknown_models), ", ".join(MODEL_NAMES)
            )
        )
    if len(set(models_run)) != len(models_run):
        raise ConfigurationError("models.run must not contain duplicate model names")
    model_paths_config = _mapping(models["configs"], "models.configs")
    _keys(model_paths_config, set(MODEL_NAMES), "models.configs")
    model_configs: Dict[str, Dict[str, Any]] = {}
    model_config_paths: Dict[str, Path] = {}
    for model_name in models_run:
        if model_name not in model_paths_config:
            raise ConfigurationError(
                "models.configs is missing selected model {!r}".format(model_name)
            )
        path_value = model_paths_config[model_name]
        if not isinstance(path_value, str) or not path_value:
            raise ConfigurationError(
                "models.configs.{} must be a non-empty path".format(model_name)
            )
        model_path = (source_path.parent / path_value).resolve()
        model_config = _validate_model_config(model_path)
        if model_config["name"] != model_name:
            raise ConfigurationError(
                "Model config {} names {!r}, but models.run selected {!r}".format(
                    model_path, model_config["name"], model_name
                )
            )
        if model_name == "constrained_kmeans" and "cluster_search" not in model_config:
            params = model_config["parameters"]
            minimum = model_config["constraints"]["min_cluster_size"]
            if params["n_clusters"] * minimum > n_rows - n_test:
                raise ConfigurationError(
                    "n_clusters * min_cluster_size exceeds the expected training row count"
                )
        elif model_name in {
            "constrained_coclustering",
            "robust_constrained_coclustering",
        }:
            params = model_config["parameters"]
            minimum = model_config["constraints"]["min_row_cluster_size"]
            if params["n_row_clusters"] * minimum > n_rows - n_test:
                raise ConfigurationError(
                    "n_row_clusters * min_row_cluster_size exceeds the expected training row count"
                )
        model_configs[model_name] = model_config
        model_config_paths[model_name] = model_path
    active_model_name = models_run[0]
    model_config = model_configs[active_model_name]
    model_path = model_config_paths[active_model_name]

    prototypes = _mapping(root["prototypes"], "prototypes")
    _keys(prototypes, {"aggregation"}, "prototypes")
    _require(prototypes, {"aggregation"}, "prototypes")
    if prototypes["aggregation"] != "median":
        raise ConfigurationError("prototypes.aggregation only supports 'median'")

    transformation = _mapping(root["transformation"], "transformation")
    _keys(transformation, {"selected_indicator", "replace_when"}, "transformation")
    _require(transformation, {"selected_indicator", "replace_when"}, "transformation")
    indicator = transformation["selected_indicator"]
    if not isinstance(indicator, str) or not re.fullmatch(
        re.escape(nonconsent["prefix"]) + r"_(0|[1-9][0-9]*)", indicator
    ):
        raise ConfigurationError(
            "transformation.selected_indicator must match <prefix>_<percentage>"
        )
    selected_percent = int(indicator.rsplit("_", 1)[1])
    if not start <= selected_percent <= end or (selected_percent - start) % step:
        raise ConfigurationError(
            "selected_indicator percentage is not one of the configured scenarios"
        )
    if _integer(transformation["replace_when"], "transformation.replace_when") not in (0, 1):
        raise ConfigurationError("transformation.replace_when must be 0 or 1")

    evaluation = _mapping(root["evaluation"], "evaluation")
    _keys(evaluation, {"enabled", "silhouette_sample_size"}, "evaluation")
    _require(evaluation, {"enabled", "silhouette_sample_size"}, "evaluation")
    if not isinstance(evaluation["enabled"], bool):
        raise ConfigurationError("evaluation.enabled must be boolean")
    _integer(evaluation["silhouette_sample_size"], "evaluation.silhouette_sample_size", 2)

    pipeline = _mapping(root["pipeline"], "pipeline")
    _keys(pipeline, {"steps"}, "pipeline")
    _require(pipeline, {"steps"}, "pipeline")
    steps = pipeline["steps"]
    if not isinstance(steps, list) or not steps:
        raise ConfigurationError("pipeline.steps must be a non-empty list")
    if (
        any(not isinstance(step, str) for step in steps)
        or len(set(steps)) != len(steps)
        or any(step not in PIPELINE_STEPS for step in steps)
    ):
        raise ConfigurationError("pipeline.steps contains unknown or duplicate stages")
    seen = set()
    for step in steps:
        missing = _DEPENDENCIES[step] - seen
        if missing:
            raise ConfigurationError(
                "pipeline stage {!r} is missing prerequisite(s): {}".format(
                    step, ", ".join(sorted(missing))
                )
            )
        seen.add(step)

    output = _mapping(root["output"], "output")
    _keys(output, {"root", "run_name", "overwrite"}, "output")
    _require(output, {"root", "run_name", "overwrite"}, "output")
    if not isinstance(output["root"], str) or not output["root"]:
        raise ConfigurationError("output.root must be a non-empty path")
    if not isinstance(output["run_name"], str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.-]*", output["run_name"]
    ):
        raise ConfigurationError("output.run_name must be a safe directory name")
    if not isinstance(output["overwrite"], bool):
        raise ConfigurationError("output.overwrite must be boolean")

    normalized = copy.deepcopy(dict(root))
    normalized["models"]["configs"] = {
        name: str(model_config_paths[name]) for name in models_run
    }
    normalized["output"]["root"] = str((source_path.parent / output["root"]).resolve())
    return LoadedConfig(
        normalized,
        source_path,
        model_config,
        model_path,
        models_run,
        model_configs,
        model_config_paths,
    )


def load_config(path: Any) -> LoadedConfig:
    """Load and validate YAML, resolving configured paths against its location."""
    source_path = Path(path).expanduser().resolve()
    try:
        with source_path.open("r", encoding="utf-8") as stream:
            contents = yaml.safe_load(stream)
    except OSError as exc:
        raise ConfigurationError("Cannot read configuration {}: {}".format(source_path, exc))
    except yaml.YAMLError as exc:
        raise ConfigurationError("Invalid YAML in {}: {}".format(source_path, exc))
    return _validate(contents, source_path)
