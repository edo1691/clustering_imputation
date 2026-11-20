"""Run-scoped artifact writing, provenance, and compatibility checks."""

import json
import hashlib
import os
import platform
import tempfile
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import joblib
import numpy
import pandas
import pyarrow
import sklearn
import yaml

from .config import LoadedConfig
from .robust_support import uses_candidate_artifacts
from .robust_support import ROBUST_MODELS, ClusterPair, search_candidates


MANIFEST_NAME = "manifest.json"


def artifact_path(config: LoadedConfig, relative_path: str) -> Path:
    return config.artifact_dir / relative_path


def stage_fingerprint(config: LoadedConfig, stage: str) -> str:
    settings: Dict[str, Any]
    if stage == "generate_dataset":
        settings = {
            "dataset": config["dataset"],
            "features": config["features"],
            "nonconsent": config["nonconsent"],
        }
    elif stage == "split_dataset":
        settings = {
            "generation": stage_fingerprint(config, "generate_dataset"),
            "split": config["split"],
        }
    elif stage == "fit_preprocessor":
        settings = {
            "split": stage_fingerprint(config, "split_dataset"),
            "preprocessing": config["preprocessing"],
            "features": config["features"],
        }
    elif stage == "train_model":
        model_settings = config.model_config
        if (
            model_settings["name"]
            in {"constrained_kmeans", "spectral_coclustering", "robust_coclustering", "robust_constrained_coclustering", "kmeans"}
            and uses_candidate_artifacts(config)
        ):
            model_settings = {
                "name": model_settings["name"],
                "parameters": {
                    key: value
                    for key, value in model_settings["parameters"].items()
                    if key not in ({"g", "m"} if model_settings["name"] in ROBUST_MODELS else {"n_clusters"})
                },
            }
            if "constraints" in config.model_config:
                model_settings["constraints"] = config.model_config["constraints"]
        settings = {
            "preprocessor": stage_fingerprint(config, "fit_preprocessor"),
            "model": model_settings,
        }
    elif stage == "build_prototypes":
        settings = {
            "training": stage_fingerprint(config, "train_model"),
            "prototypes": config["prototypes"],
        }
    elif stage == "predict_test_clusters":
        settings = {
            "split": stage_fingerprint(config, "split_dataset"),
            "preprocessor": stage_fingerprint(config, "fit_preprocessor"),
            "model": stage_fingerprint(config, "train_model"),
            "selected_candidate": selected_candidate(config),
        }
    elif stage == "transform_test":
        settings = {
            "prediction": stage_fingerprint(config, "predict_test_clusters"),
            "prototypes": stage_fingerprint(config, "build_prototypes"),
            "nonconsent": config["nonconsent"],
            "replace_when": config["transformation"]["replace_when"],
        }
    elif stage == "evaluate":
        settings = {
            "training": stage_fingerprint(config, "train_model"),
            "prediction": stage_fingerprint(config, "predict_test_clusters"),
            "prototypes": stage_fingerprint(config, "build_prototypes"),
            "transformation": stage_fingerprint(config, "transform_test"),
            "evaluation": config["evaluation"],
        }
    else:
        raise ValueError("Unknown stage fingerprint: {}".format(stage))
    payload = json.dumps(settings, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def selected_candidate(config: LoadedConfig) -> Optional[int]:
    search = config.model_config.get("cluster_search")
    if (
        config.model_config["name"]
        in {"constrained_kmeans", "spectral_coclustering", "robust_coclustering", "robust_constrained_coclustering", "kmeans"}
        and search
        and uses_candidate_artifacts(config)
    ):
        selection_path = selection_artifact_path(config)
        if not artifact_path(config, selection_path).is_file():
            configured_k = search["selected_n_clusters"]
            if isinstance(configured_k, int) and not isinstance(configured_k, bool):
                try:
                    candidate = require_stage(
                        config,
                        candidate_stage_key(
                            "train_model",
                            configured_k,
                            config.model_config["name"],
                        ),
                        candidate_training_fingerprint(config, configured_k),
                        [],
                    ).get("metadata", {})
                except FileNotFoundError:
                    candidate = {}
                if candidate.get("status") not in (None, "success"):
                    raise ValueError(
                        "Selected cluster count {} was not successfully trained "
                        "(status {}: {}).".format(
                            configured_k,
                            candidate.get("status"),
                            candidate.get("failure_reason") or "no failure reason recorded",
                        )
                    )
            raise FileNotFoundError(
                "Cluster selection has not been resolved; run train_model first "
                "for configured selection {!r}.".format(configured_k)
            )
        selection = read_json(config, selection_path)
        if config.model_config["name"] in ROBUST_MODELS:
            pair = selection["selected_pair"]
            return ClusterPair(pair["g"], pair["m"])
        return int(selection["selected_n_clusters"])
    return None


def candidate_training_fingerprint(config: LoadedConfig, n_clusters: int) -> str:
    model = config.model_config
    settings = {
        "preprocessor": stage_fingerprint(config, "fit_preprocessor"),
        "model": model["name"],
        "parameters": {
            key: value
            for key, value in model["parameters"].items()
            if key not in ({"g", "m"} if model["name"] in ROBUST_MODELS else {"n_clusters", "cluster_search"})
        },
        "constraints": model.get("constraints", "not_applicable"),
        "n_clusters": n_clusters,
    }
    if model["name"] in ROBUST_MODELS:
        settings["candidate_schema"] = "robust_pairs_v1"
    return _fingerprint(settings)


def candidate_prototype_fingerprint(config: LoadedConfig, n_clusters: int) -> str:
    return _fingerprint(
        {
            "training_candidate": candidate_training_fingerprint(config, n_clusters),
            "prototypes": config["prototypes"],
        }
    )


def candidate_stage_key(
    stage: str, n_clusters: int, model_name: str = "constrained_kmeans"
) -> str:
    return "{}:{}:k_{}".format(stage, model_name, n_clusters)


def cluster_search_fingerprint(config: LoadedConfig) -> str:
    search = config.model_config.get("cluster_search")
    if not search:
        raise ValueError("This configuration does not enable cluster search")
    return _fingerprint(
        {
            "model": stage_fingerprint(config, "train_model"),
            "candidates": search_candidates(config),
        }
    )


def selection_fingerprint(config: LoadedConfig) -> str:
    search = config.model_config.get("cluster_search")
    if not uses_candidate_artifacts(config):
        raise ValueError("Cluster selection artifacts require enabled cluster search")
    settings = {
        "cluster_search": cluster_search_fingerprint(config),
        "configured_selected_n_clusters": search["selected_n_clusters"],
    }
    if config.model_config["name"] in ROBUST_MODELS:
        settings["selection_schema_version"] = 2
    return _fingerprint(settings)


def selection_stage_key(config: LoadedConfig) -> str:
    return "select_model:{}".format(selection_fingerprint(config)[:12])


def selection_artifact_path(config: LoadedConfig) -> str:
    return "models/{}/selection_{}.json".format(
        config.model_config["name"], selection_fingerprint(config)[:12]
    )


def _fingerprint(settings: Mapping[str, Any]) -> str:
    payload = json.dumps(settings, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def initialize_run(config: LoadedConfig) -> Dict[str, Any]:
    root = config.artifact_dir
    manifest_path = root / MANIFEST_NAME
    if root.exists() and not manifest_path.exists():
        existing = [path for path in root.iterdir()]
        if existing and not config["output"]["overwrite"]:
            raise FileExistsError(
                "Run directory exists without a manifest: {}. Choose another "
                "output.run_name or set output.overwrite=true.".format(root)
            )
    root.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as stream:
            manifest = json.load(stream)
    else:
        manifest = {
            "schema_version": 1,
            "run_name": config["output"]["run_name"],
            "created_at": _timestamp(),
            "stages": {},
        }
    manifest["updated_at"] = _timestamp()
    manifest["config_fingerprint"] = config.fingerprint
    _write_yaml_metadata(root / "resolved_config.yml", _serializable_config(config))
    environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "pyarrow": pyarrow.__version__,
        "joblib": joblib.__version__,
        "scikit_learn": sklearn.__version__,
        "k_means_constrained": version("k-means-constrained"),
        "ortools": version("ortools"),
        "pyyaml": version("PyYAML"),
        "config_fingerprint": config.fingerprint,
    }
    _write_json_metadata(root / "environment.json", environment)
    _write_manifest(config, manifest)
    return manifest


def require_stage(
    config: LoadedConfig,
    stage_key: str,
    expected_fingerprint: str,
    required_artifacts: Iterable[str],
) -> Dict[str, Any]:
    manifest_path = artifact_path(config, MANIFEST_NAME)
    if not manifest_path.exists():
        raise FileNotFoundError(
            "Required artifacts are missing; run generate_dataset first."
        )
    with manifest_path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    record = manifest.get("stages", {}).get(stage_key)
    if record is None:
        raise FileNotFoundError(
            "Prerequisite stage {!r} has not completed; run it first.".format(stage_key)
        )
    if record.get("fingerprint") != expected_fingerprint:
        raise ValueError(
            "Artifacts from stage {!r} are incompatible with this configuration; "
            "rerun that stage.".format(stage_key)
        )
    for relative in required_artifacts:
        if not artifact_path(config, relative).is_file():
            raise FileNotFoundError(
                "Stage {!r} artifact {} is missing; rerun that stage.".format(
                    stage_key, relative
                )
            )
    return record


def record_stage(
    config: LoadedConfig,
    stage_key: str,
    fingerprint: str,
    artifacts: Iterable[str],
    dependencies: Optional[Mapping[str, str]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> None:
    manifest = initialize_run(config)
    previous = manifest["stages"].get(stage_key)
    if previous is not None and not config["output"]["overwrite"]:
        raise FileExistsError(
            "Stage {!r} already has artifacts; set output.overwrite=true or "
            "use a new run_name.".format(stage_key)
        )
    manifest["stages"][stage_key] = {
        "fingerprint": fingerprint,
        "artifacts": list(artifacts),
        "dependencies": dict(dependencies or {}),
        "metadata": dict(metadata or {}),
        "completed_at": _timestamp(),
    }
    _write_manifest(config, manifest)


def write_frame(
    config: LoadedConfig,
    relative: str,
    frame: pandas.DataFrame,
    overwrite_existing: bool = False,
) -> Path:
    target = artifact_path(config, relative)
    _prepare_target(config, target, allow_existing=overwrite_existing)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(str(temporary), str(target))
    return target


def write_csv(
    config: LoadedConfig,
    relative: str,
    frame: pandas.DataFrame,
    overwrite_existing: bool = False,
) -> Path:
    target = artifact_path(config, relative)
    _prepare_target(config, target, allow_existing=overwrite_existing)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    try:
        frame.to_csv(temporary, index=False)
        os.replace(str(temporary), str(target))
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise
    return target


def write_text(
    config: LoadedConfig,
    relative: str,
    content: str,
    overwrite_existing: bool = False,
) -> Path:
    target = artifact_path(config, relative)
    _prepare_target(config, target, allow_existing=overwrite_existing)
    target.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(target, content)
    return target


def read_frame(config: LoadedConfig, relative: str) -> pandas.DataFrame:
    target = artifact_path(config, relative)
    if not target.is_file():
        raise FileNotFoundError(
            "Missing artifact {} in run {}; run its producing stage first.".format(
                relative, config.artifact_dir
            )
        )
    return pandas.read_parquet(target)


def write_object(config: LoadedConfig, relative: str, value: Any) -> Path:
    target = artifact_path(config, relative)
    _prepare_target(config, target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    joblib.dump(value, temporary)
    os.replace(str(temporary), str(target))
    return target


def read_object(config: LoadedConfig, relative: str) -> Any:
    target = artifact_path(config, relative)
    if not target.is_file():
        raise FileNotFoundError(
            "Missing artifact {}; run its producing stage first.".format(relative)
        )
    return joblib.load(target)


def write_json(config: LoadedConfig, relative: str, value: Any) -> Path:
    target = artifact_path(config, relative)
    _prepare_target(config, target)
    target.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(target, json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    return target


def read_json(config: LoadedConfig, relative: str) -> Any:
    target = artifact_path(config, relative)
    if not target.is_file():
        raise FileNotFoundError(
            "Missing artifact {}; run its producing stage first.".format(relative)
        )
    with target.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def file_sizes(config: LoadedConfig, relative_paths: Iterable[str]) -> Dict[str, int]:
    return {
        relative: artifact_path(config, relative).stat().st_size
        for relative in relative_paths
        if artifact_path(config, relative).is_file()
    }


def _prepare_target(
    config: LoadedConfig, target: Path, allow_existing: bool = False
) -> None:
    if target.exists() and not config["output"]["overwrite"] and not allow_existing:
        raise FileExistsError(
            "Refusing to overwrite artifact {}. Set output.overwrite=true or use "
            "a new run_name.".format(target)
        )


def _write_manifest(config: LoadedConfig, manifest: Mapping[str, Any]) -> None:
    target = artifact_path(config, MANIFEST_NAME)
    target.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(target, json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def _write_json_metadata(target: Path, value: Mapping[str, Any]) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(target, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _write_yaml_metadata(target: Path, value: Mapping[str, Any]) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(target, yaml.safe_dump(value, sort_keys=False))


def _atomic_write(target: Path, content: str) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=target.name + ".", suffix=".tmp", dir=str(target.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(content)
        os.replace(temporary_name, str(target))
    except Exception:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
        raise


def _serializable_config(config: LoadedConfig) -> Dict[str, Any]:
    settings = json.loads(json.dumps(dict(config), default=str))
    settings["resolved_model_configs"] = config.model_configs
    settings["resolved_model_config_paths"] = {
        name: str(path) for name, path in config.model_config_paths.items()
    }
    settings["models"]["run"] = list(config.models_run)
    return settings


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()
