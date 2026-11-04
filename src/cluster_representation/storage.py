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
        settings = {
            "preprocessor": stage_fingerprint(config, "fit_preprocessor"),
            "model": config.model_config,
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
        }
    elif stage == "transform_test":
        settings = {
            "prediction": stage_fingerprint(config, "predict_test_clusters"),
            "prototypes": stage_fingerprint(config, "build_prototypes"),
            "indicator": config["transformation"]["selected_indicator"],
            "replace_when": config["transformation"]["replace_when"],
        }
    elif stage == "evaluate":
        settings = {
            "training": stage_fingerprint(config, "train_model"),
            "prediction": stage_fingerprint(config, "predict_test_clusters"),
            "prototypes": stage_fingerprint(config, "build_prototypes"),
            "evaluation": config["evaluation"],
            "replace_when": config["transformation"]["replace_when"],
        }
    else:
        raise ValueError("Unknown stage fingerprint: {}".format(stage))
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


def write_frame(config: LoadedConfig, relative: str, frame: pandas.DataFrame) -> Path:
    target = artifact_path(config, relative)
    _prepare_target(config, target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(str(temporary), str(target))
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


def _prepare_target(config: LoadedConfig, target: Path) -> None:
    if target.exists() and not config["output"]["overwrite"]:
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
    return settings


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()
