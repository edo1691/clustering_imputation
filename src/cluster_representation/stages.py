"""Public, artifact-backed workflow stages shared by CLI and notebooks."""

from dataclasses import dataclass
import json
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from . import evaluation
from . import generation, nonconsent, preprocessing, prototypes, reporting, splitting
from . import storage, transformation
from .config import LoadedConfig
from .models import create_model
from .randomness import generator
from .validation import validate_stage_artifacts


@dataclass
class StageResult:
    stage: str
    artifacts: Dict[str, str]
    summary: Dict[str, Any]
    config: LoadedConfig

    def preview(self, artifact: Optional[str] = None, rows: int = 5) -> Any:
        """Display a small artifact preview, or the result summary."""
        if artifact is None:
            return self.summary
        if artifact not in self.artifacts:
            raise KeyError("Unknown artifact key {!r}".format(artifact))
        path = self.config.artifact_dir / self.artifacts[artifact]
        if path.suffix == ".parquet":
            return pd.read_parquet(path).head(rows)
        if path.suffix == ".json":
            return storage.read_json(self.config, self.artifacts[artifact])
        return str(path)

    def report(self) -> Any:
        comparison_path = self.artifacts.get("comparison")
        if comparison_path is not None:
            return reporting.cluster_search_report(
                storage.read_frame(self.config, comparison_path)
            )
        metrics_path = self.artifacts.get("metrics")
        if metrics_path is None:
            return self.summary
        return reporting.display_report(storage.read_json(self.config, metrics_path))


def generate_dataset(config: LoadedConfig) -> StageResult:
    config = _require_config(config)
    storage.initialize_run(config)
    features = generation.generate_features(config)
    indicators = nonconsent.generate_nonconsent(config)
    feature_path = "data/features.parquet"
    indicator_path = "data/nonconsent.parquet"
    storage.write_frame(config, feature_path, features)
    storage.write_frame(config, indicator_path, indicators)
    fingerprint = storage.stage_fingerprint(config, "generate_dataset")
    storage.record_stage(
        config,
        "generate_dataset",
        fingerprint,
        [feature_path, indicator_path],
        metadata={
            "n_rows": len(features),
            "feature_columns": list(features.columns),
            "nonconsent_columns": list(indicators.columns),
            "scenario_counts": {
                column: int(indicators[column].sum())
                for column in indicators.columns[1:]
            },
            "seed": config["dataset"]["seed"],
        },
    )
    return StageResult(
        "generate_dataset",
        {"features": feature_path, "nonconsent": indicator_path},
        {"n_rows": len(features), "n_features": len(config["features"])},
        config,
    )


def split_dataset(config: LoadedConfig) -> StageResult:
    config = _require_config(config)
    input_paths = ["data/features.parquet", "data/nonconsent.parquet"]
    generation_fp = storage.stage_fingerprint(config, "generate_dataset")
    validate_stage_artifacts(config, "generate_dataset", generation_fp, input_paths)
    result = splitting.split_data(
        storage.read_frame(config, input_paths[0]),
        storage.read_frame(config, input_paths[1]),
        config,
    )
    paths = {
        "train_features": "data/train_features.parquet",
        "test_features": "data/test_features.parquet",
        "train_indicators": "data/train_indicators.parquet",
        "test_indicators": "data/test_indicators.parquet",
    }
    for key, path in paths.items():
        storage.write_frame(config, path, result[key])
    fingerprint = storage.stage_fingerprint(config, "split_dataset")
    storage.record_stage(
        config,
        "split_dataset",
        fingerprint,
        list(paths.values()),
        {"generate_dataset": generation_fp},
        dict(result["summary"], seed=config["split"]["seed"]),
    )
    return StageResult("split_dataset", paths, result["summary"], config)


def fit_preprocessor(config: LoadedConfig) -> StageResult:
    config = _require_config(config)
    split_fp = storage.stage_fingerprint(config, "split_dataset")
    train_path = "data/train_features.parquet"
    validate_stage_artifacts(config, "split_dataset", split_fp, [train_path])
    train_features = storage.read_frame(config, train_path)
    id_column = config["dataset"]["id_column"]
    names = [feature["name"] for feature in config["features"]]
    fitted = preprocessing.fit_preprocessor(train_features[names], config)
    model_path = "preprocessing/preprocessor.joblib"
    schema_path = "preprocessing/schema.json"
    schema = {
        "id_column": id_column,
        "feature_order": names,
        "input_columns": names,
        "training_dtypes": {
            column: str(train_features[column].dtype) for column in names
        },
        "imputation": config["preprocessing"]["imputation"],
        "scaling": config["preprocessing"]["scaling"],
        "training_id_count": len(train_features),
        "training_id_fingerprint": _ids_fingerprint(train_features[id_column]),
    }
    storage.write_object(config, model_path, fitted)
    storage.write_json(config, schema_path, schema)
    fingerprint = storage.stage_fingerprint(config, "fit_preprocessor")
    storage.record_stage(
        config,
        "fit_preprocessor",
        fingerprint,
        [model_path, schema_path],
        {"split_dataset": split_fp},
        {"feature_order": names, "training_id_count": len(train_features)},
    )
    return StageResult(
        "fit_preprocessor",
        {"preprocessor": model_path, "schema": schema_path},
        {"feature_order": names, "training_id_count": len(train_features)},
        config,
    )


def train_model(config: LoadedConfig) -> StageResult:
    config = _require_config(config)
    preprocessor_fp = storage.stage_fingerprint(config, "fit_preprocessor")
    preprocessor_path = "preprocessing/preprocessor.joblib"
    train_path = "data/train_features.parquet"
    validate_stage_artifacts(
        config,
        "fit_preprocessor",
        preprocessor_fp,
        [preprocessor_path, "preprocessing/schema.json", train_path],
    )
    train_features = storage.read_frame(config, train_path)
    names = [feature["name"] for feature in config["features"]]
    fitted_preprocessor = storage.read_object(config, preprocessor_path)
    schema = storage.read_json(config, "preprocessing/schema.json")
    if schema["feature_order"] != names:
        raise ValueError("Saved preprocessor feature order is incompatible with configuration")
    if schema["training_id_fingerprint"] != _ids_fingerprint(
        train_features[config["dataset"]["id_column"]]
    ):
        raise ValueError("Saved preprocessor was fitted on different training IDs")
    if config.model_config.get("cluster_search", {}).get("enabled", False):
        return _train_cluster_search(
            config,
            train_features,
            fitted_preprocessor.transform(train_features[names]),
            preprocessor_fp,
        )

    model_config = dict(config.model_config)
    model_config["parameters"] = dict(config.model_config["parameters"])
    model_config.pop("cluster_search", None)
    model = create_model(model_config)
    fit_started = time.perf_counter()
    model.fit(fitted_preprocessor.transform(train_features[names]))
    fit_seconds = time.perf_counter() - fit_started
    model_path = "models/model.joblib"
    metadata_path = "models/metadata.json"
    storage.write_object(config, model_path, model)
    training_labels = np.asarray(model.labels_, dtype=int)
    metadata = model.metadata()
    metadata["training_id_fingerprint"] = _ids_fingerprint(
        train_features[config["dataset"]["id_column"]]
    )
    metadata["feature_order"] = names
    metadata["fit_seconds"] = fit_seconds
    metadata["training_cluster_sizes"] = {
        str(int(cluster)): int(count)
        for cluster, count in zip(*np.unique(training_labels, return_counts=True))
    }
    storage.write_json(config, metadata_path, metadata)
    fingerprint = storage.stage_fingerprint(config, "train_model")
    storage.record_stage(
        config,
        "train_model",
        fingerprint,
        [model_path, metadata_path],
        {"fit_preprocessor": preprocessor_fp},
        {
            "training_rows": len(train_features),
            "model": metadata["name"],
            "fit_seconds": fit_seconds,
        },
    )
    return StageResult(
        "train_model",
        {"model": model_path, "metadata": metadata_path},
        {"training_rows": len(train_features), "cluster_sizes": metadata["training_cluster_sizes"]},
        config,
    )


def _train_cluster_search(
    config: LoadedConfig,
    train_features: pd.DataFrame,
    train_inputs: np.ndarray,
    preprocessor_fingerprint: str,
) -> StageResult:
    search = config.model_config["cluster_search"]
    n_train = len(train_features)
    id_column = config["dataset"]["id_column"]
    seed = config.model_config["parameters"]["random_state"]
    sample_size = min(n_train, config["evaluation"]["silhouette_sample_size"])
    sample_rng = generator(seed, "cluster-search:silhouette")
    silhouette_indices = np.sort(
        sample_rng.choice(n_train, size=sample_size, replace=False)
    )
    rows = []
    candidate_artifacts: Dict[str, str] = {}
    minimum_size = config.model_config["constraints"]["min_cluster_size"]

    for candidate in search["candidates"]:
        base = "models/constrained_kmeans/k_{}".format(candidate)
        paths = {
            "model": base + "/model.joblib",
            "assignments": base + "/train_cluster_assignments.parquet",
            "metadata": base + "/metadata.json",
        }
        stage_key = storage.candidate_stage_key("train_model", candidate)
        fingerprint = storage.candidate_training_fingerprint(config, candidate)
        candidate_exists = False
        try:
            record = storage.require_stage(config, stage_key, fingerprint, [])
            candidate_result = dict(record.get("metadata", {}))
            candidate_exists = True
        except FileNotFoundError:
            candidate_result = {}
        if candidate_exists:
            if candidate_result.get("status") == "success":
                storage.require_stage(
                    config, stage_key, fingerprint, paths.values()
                )
        else:
            if candidate * minimum_size > n_train:
                candidate_result = _candidate_result(
                    candidate,
                    status="skipped_infeasible",
                    failure_reason=(
                        "{} * {} exceeds n_train={}".format(
                            candidate, minimum_size, n_train
                        )
                    ),
                )
                storage.record_stage(
                    config,
                    stage_key,
                    fingerprint,
                    [],
                    {"fit_preprocessor": preprocessor_fingerprint},
                    candidate_result,
                )
            else:
                candidate_model_config = {
                    "name": "constrained_kmeans",
                    "parameters": dict(config.model_config["parameters"]),
                    "constraints": config.model_config["constraints"],
                }
                candidate_model_config["parameters"]["n_clusters"] = candidate
                model = create_model(candidate_model_config)
                started = time.perf_counter()
                try:
                    model.fit(train_inputs)
                except (ValueError, RuntimeError) as exc:
                    training_seconds = time.perf_counter() - started
                    candidate_result = _candidate_result(
                        candidate,
                        status="failed",
                        failure_reason="{}: {}".format(type(exc).__name__, exc),
                        training_time_seconds=training_seconds,
                    )
                    storage.record_stage(
                        config,
                        stage_key,
                        fingerprint,
                        [],
                        {"fit_preprocessor": preprocessor_fingerprint},
                        candidate_result,
                    )
                else:
                    training_seconds = time.perf_counter() - started
                    labels = np.asarray(model.labels_, dtype=int)
                    counts = np.bincount(labels, minlength=candidate + 1)[1:]
                    if len(counts) != candidate:
                        raise RuntimeError(
                            "Candidate {} returned an unexpected number of clusters".format(
                                candidate
                            )
                        )
                    violations = np.flatnonzero(counts < minimum_size) + 1
                    metrics = evaluation.evaluate_training_candidate(
                        train_inputs, labels, silhouette_indices
                    )
                    candidate_result = _candidate_result(
                        candidate,
                        status="success",
                        failure_reason=None,
                        training_time_seconds=training_seconds,
                        constraint_violations=int(len(violations)),
                        constraint_violation_clusters=violations.tolist(),
                        training_id_fingerprint=_ids_fingerprint(
                            train_features[id_column]
                        ),
                        feature_order=[feature["name"] for feature in config["features"]],
                        model_parameters=json.dumps(
                            candidate_model_config["parameters"], sort_keys=True
                        ),
                        constraints=json.dumps(
                            candidate_model_config["constraints"], sort_keys=True
                        ),
                        **metrics
                    )
                    assignments = pd.DataFrame(
                        {
                            id_column: train_features[id_column].to_numpy(),
                            "cluster": labels,
                        }
                    )
                    storage.write_object(config, paths["model"], model)
                    storage.write_frame(config, paths["assignments"], assignments)
                    storage.write_json(config, paths["metadata"], candidate_result)
                    storage.record_stage(
                        config,
                        stage_key,
                        fingerprint,
                        list(paths.values()),
                        {"fit_preprocessor": preprocessor_fingerprint},
                        candidate_result,
                    )
        rows.append(candidate_result)
        if candidate_result["status"] == "success":
            candidate_artifacts["model_k_{}".format(candidate)] = paths["model"]
            candidate_artifacts["assignments_k_{}".format(candidate)] = paths[
                "assignments"
            ]

    comparison = pd.DataFrame(rows)
    search_fingerprint = storage.cluster_search_fingerprint(config)
    comparison_path = "models/constrained_kmeans/search_{}/comparison.parquet".format(
        search_fingerprint[:12]
    )
    search_stage_key = "train_model:search:{}".format(search_fingerprint[:12])
    try:
        storage.require_stage(
            config,
            search_stage_key,
            search_fingerprint,
            [comparison_path],
        )
    except FileNotFoundError:
        storage.write_frame(config, comparison_path, comparison)
        storage.record_stage(
            config,
            search_stage_key,
            search_fingerprint,
            [comparison_path],
            {
                storage.candidate_stage_key("train_model", candidate):
                    storage.candidate_training_fingerprint(config, candidate)
                for candidate in search["candidates"]
            },
            {"candidate_count": len(search["candidates"])},
        )
    else:
        comparison = storage.read_frame(config, comparison_path)
    candidate_artifacts["comparison"] = comparison_path
    selected_row = comparison.loc[
        comparison["n_clusters"].eq(search["selected_n_clusters"])
    ].iloc[0]
    return StageResult(
        "train_model",
        candidate_artifacts,
        {
            "training_rows": n_train,
            "candidate_count": len(search["candidates"]),
            "selected_n_clusters": search["selected_n_clusters"],
            "selected_status": selected_row["status"],
            "candidates": rows,
        },
        config,
    )


def _candidate_result(
    n_clusters: int,
    status: str,
    failure_reason: Optional[str],
    training_time_seconds: Optional[float] = None,
    minimum_cluster_size: Optional[int] = None,
    maximum_cluster_size: Optional[int] = None,
    constraint_violations: Optional[int] = None,
    constraint_violation_clusters: Optional[list] = None,
    **metrics: Any
) -> Dict[str, Any]:
    result = {
        "n_clusters": n_clusters,
        "status": status,
        "failure_reason": failure_reason,
        "minimum_cluster_size": minimum_cluster_size,
        "maximum_cluster_size": maximum_cluster_size,
        "constraint_violations": constraint_violations,
        "constraint_violation_clusters": constraint_violation_clusters or [],
        "training_time_seconds": training_time_seconds,
        "silhouette_score": None,
        "silhouette_status": "not_evaluated",
        "davies_bouldin_index": None,
        "davies_bouldin_status": "not_evaluated",
        "calinski_harabasz_index": None,
        "calinski_harabasz_status": "not_evaluated",
    }
    result.update(metrics)
    return result


def _candidate_training_paths(n_clusters: int) -> Tuple[str, str, str]:
    base = "models/constrained_kmeans/k_{}".format(n_clusters)
    return (
        base + "/model.joblib",
        base + "/train_cluster_assignments.parquet",
        base + "/metadata.json",
    )


def _candidate_prototype_path(n_clusters: int) -> str:
    return "models/constrained_kmeans/k_{}/cluster_prototypes.parquet".format(
        n_clusters
    )


def _cluster_search_comparison_path(config: LoadedConfig) -> str:
    fingerprint = storage.cluster_search_fingerprint(config)
    return "models/constrained_kmeans/search_{}/comparison.parquet".format(
        fingerprint[:12]
    )


def _cluster_search_stage_key(config: LoadedConfig) -> str:
    fingerprint = storage.cluster_search_fingerprint(config)
    return "train_model:search:{}".format(fingerprint[:12])


def _require_selected_candidate(config: LoadedConfig, n_clusters: int) -> tuple:
    model_path, assignments_path, metadata_path = _candidate_training_paths(n_clusters)
    key = storage.candidate_stage_key("train_model", n_clusters)
    fingerprint = storage.candidate_training_fingerprint(config, n_clusters)
    record = storage.require_stage(config, key, fingerprint, [])
    metadata = record.get("metadata", {})
    if metadata.get("status") != "success":
        raise ValueError(
            "Selected cluster count {} was not successfully trained ({}: {}).".format(
                n_clusters,
                metadata.get("status", "missing status"),
                metadata.get("failure_reason") or "no failure details",
            )
        )
    storage.require_stage(
        config, key, fingerprint, [model_path, assignments_path, metadata_path]
    )
    return model_path, assignments_path, metadata_path


def _require_selected_candidate_prototypes(
    config: LoadedConfig, n_clusters: int
) -> str:
    path = _candidate_prototype_path(n_clusters)
    key = storage.candidate_stage_key("build_prototypes", n_clusters)
    fingerprint = storage.candidate_prototype_fingerprint(config, n_clusters)
    storage.require_stage(config, key, fingerprint, [path])
    return path


def build_prototypes(config: LoadedConfig) -> StageResult:
    config = _require_config(config)
    if config.model_config.get("cluster_search", {}).get("enabled", False):
        return _build_cluster_search_prototypes(config)
    train_fp = storage.stage_fingerprint(config, "train_model")
    model_path = "models/model.joblib"
    metadata_path = "models/metadata.json"
    train_path = "data/train_features.parquet"
    validate_stage_artifacts(
        config, "train_model", train_fp, [model_path, metadata_path, train_path]
    )
    model = storage.read_object(config, model_path)
    train = storage.read_frame(config, train_path)
    model_metadata = storage.read_json(config, metadata_path)
    if model_metadata["training_id_fingerprint"] != _ids_fingerprint(
        train[config["dataset"]["id_column"]]
    ):
        raise ValueError("Saved model was fitted on different training IDs")
    minimum = config.model_config.get("constraints", {}).get("min_cluster_size", 1)
    built = prototypes.build_prototypes(train, model.labels_, config, minimum)
    assignment_path = "prototypes/train_cluster_assignments.parquet"
    prototype_path = "prototypes/cluster_prototypes.parquet"
    storage.write_frame(config, assignment_path, built["assignments"])
    storage.write_frame(config, prototype_path, built["prototypes"])
    fingerprint = storage.stage_fingerprint(config, "build_prototypes")
    storage.record_stage(
        config,
        "build_prototypes",
        fingerprint,
        [assignment_path, prototype_path],
        {"train_model": train_fp},
        {
            "n_clusters": len(built["prototypes"]),
            "support": built["prototypes"]["n_clients"].astype(int).tolist(),
            "minimum_cluster_size": minimum,
        },
    )
    return StageResult(
        "build_prototypes",
        {"assignments": assignment_path, "prototypes": prototype_path},
        {
            "n_clusters": len(built["prototypes"]),
            "minimum_support": int(built["prototypes"]["n_clients"].min()),
        },
        config,
    )


def _build_cluster_search_prototypes(config: LoadedConfig) -> StageResult:
    search = config.model_config["cluster_search"]
    train_path = "data/train_features.parquet"
    split_fingerprint = storage.stage_fingerprint(config, "split_dataset")
    validate_stage_artifacts(config, "split_dataset", split_fingerprint, [train_path])
    train_features = storage.read_frame(config, train_path)
    comparison_path = _cluster_search_comparison_path(config)
    search_fingerprint = storage.cluster_search_fingerprint(config)
    storage.require_stage(
        config,
        _cluster_search_stage_key(config),
        search_fingerprint,
        [comparison_path],
    )
    comparison = storage.read_frame(config, comparison_path)
    minimum_size = config.model_config["constraints"]["min_cluster_size"]
    artifacts: Dict[str, str] = {"comparison": comparison_path}
    candidate_summaries = []

    for candidate_row in comparison.to_dict(orient="records"):
        candidate = int(candidate_row["n_clusters"])
        if candidate_row["status"] != "success":
            continue
        model_path, assignment_path, metadata_path = _candidate_training_paths(candidate)
        candidate_train_key = storage.candidate_stage_key("train_model", candidate)
        candidate_train_fingerprint = storage.candidate_training_fingerprint(
            config, candidate
        )
        training_record = storage.require_stage(
            config,
            candidate_train_key,
            candidate_train_fingerprint,
            [model_path, assignment_path, metadata_path],
        )
        if training_record["metadata"].get("status") != "success":
            raise ValueError(
                "Candidate {} has no successful fitted training artifacts".format(candidate)
            )
        model = storage.read_object(config, model_path)
        saved_assignments = storage.read_frame(config, assignment_path)
        expected_ids = train_features[config["dataset"]["id_column"]].tolist()
        if saved_assignments[config["dataset"]["id_column"]].tolist() != expected_ids:
            raise ValueError(
                "Candidate {} training assignments do not align with training IDs".format(
                    candidate
                )
            )
        fitted_labels = np.asarray(model.labels_, dtype=int)
        if not np.array_equal(fitted_labels, saved_assignments["cluster"].to_numpy()):
            raise ValueError(
                "Candidate {} saved assignments differ from fitted training labels".format(
                    candidate
                )
            )
        prototype_path = _candidate_prototype_path(candidate)
        prototype_key = storage.candidate_stage_key("build_prototypes", candidate)
        prototype_fingerprint = storage.candidate_prototype_fingerprint(
            config, candidate
        )
        prototype_exists = False
        try:
            storage.require_stage(
                config, prototype_key, prototype_fingerprint, []
            )
            prototype_exists = True
        except FileNotFoundError:
            pass
        if prototype_exists:
            storage.require_stage(
                config, prototype_key, prototype_fingerprint, [prototype_path]
            )
            prototype_frame = storage.read_frame(config, prototype_path)
        else:
            built = prototypes.build_prototypes(
                train_features, fitted_labels, config, minimum_size
            )
            prototype_frame = built["prototypes"]
            storage.write_frame(config, prototype_path, prototype_frame)
            storage.record_stage(
                config,
                prototype_key,
                prototype_fingerprint,
                [prototype_path],
                {candidate_train_key: candidate_train_fingerprint},
                {
                    "n_clusters": len(prototype_frame),
                    "support": prototype_frame["n_clients"].astype(int).tolist(),
                    "minimum_cluster_size": minimum_size,
                },
            )
        artifacts["prototypes_k_{}".format(candidate)] = prototype_path
        candidate_summaries.append(
            {
                "n_clusters": candidate,
                "minimum_support": int(prototype_frame["n_clients"].min()),
            }
        )

    selected = search["selected_n_clusters"]
    selected_prototype_key = "prototypes_k_{}".format(selected)
    if selected_prototype_key not in artifacts:
        selected_row = comparison.loc[comparison["n_clusters"].eq(selected)].iloc[0]
        raise ValueError(
            "Selected cluster count {} was not successfully trained ({}: {}).".format(
                selected,
                selected_row["status"],
                selected_row["failure_reason"] or "no failure details",
            )
        )
    artifacts["prototypes"] = artifacts[selected_prototype_key]
    artifacts["assignments"] = _candidate_training_paths(selected)[1]
    return StageResult(
        "build_prototypes",
        artifacts,
        {
            "candidate_prototypes": candidate_summaries,
            "selected_n_clusters": search["selected_n_clusters"],
        },
        config,
    )


def transform_test(config: LoadedConfig) -> StageResult:
    config = _require_config(config)
    transformed = _run_transformation(config, config["transformation"]["selected_indicator"])
    return transformed


def _run_transformation(config: LoadedConfig, indicator_name: str) -> StageResult:
    selected_k = storage.selected_candidate(config)
    fingerprint = storage.stage_fingerprint(config, "transform_test")
    if selected_k is None:
        stage_key = "transform_test:{}".format(indicator_name)
        prediction_stage_key = "predict_test_clusters"
        scenario_dir = "transformed/{}".format(indicator_name)
        prediction_path = "predictions/test_cluster_assignments.parquet"
    else:
        stage_key = "transform_test:k_{}:{}".format(selected_k, indicator_name)
        scenario_dir = "transformed/constrained_kmeans/k_{}/{}".format(
            selected_k, indicator_name
        )
        prediction_stage_key = "predict_test_clusters:k_{}".format(selected_k)
        prediction_path = (
            "predictions/constrained_kmeans/k_{}/test_cluster_assignments.parquet".format(
                selected_k
            )
        )
    transformed_path = scenario_dir + "/transformed_test.parquet"
    original_path = scenario_dir + "/original_test.parquet"
    manifest_path = storage.artifact_path(config, storage.MANIFEST_NAME)
    if manifest_path.exists():
        manifest = storage.initialize_run(config)
        if stage_key in manifest.get("stages", {}):
            validate_stage_artifacts(
                config,
                stage_key,
                fingerprint,
                [transformed_path, original_path],
            )
            frame = storage.read_frame(config, transformed_path)
            return StageResult(
                "transform_test",
                {
                    "transformed": transformed_path,
                    "original": original_path,
                    "test_assignments": prediction_path,
                },
                {
                    "indicator": indicator_name,
                    "test_rows": len(frame),
                    "n_replaced": int(
                        frame["gdpr"].eq(config["transformation"]["replace_when"]).sum()
                    ),
                },
                config,
            )
    dependencies = {
        "split_dataset": storage.stage_fingerprint(config, "split_dataset"),
        "fit_preprocessor": storage.stage_fingerprint(config, "fit_preprocessor"),
        "train_model": storage.stage_fingerprint(config, "train_model"),
        "build_prototypes": storage.stage_fingerprint(config, "build_prototypes"),
    }
    validate_stage_artifacts(
        config, "split_dataset", dependencies["split_dataset"], ["data/test_features.parquet"]
    )
    validate_stage_artifacts(
        config, "fit_preprocessor", dependencies["fit_preprocessor"],
        ["preprocessing/preprocessor.joblib"],
    )
    if selected_k is None:
        validate_stage_artifacts(
            config, "train_model", dependencies["train_model"], ["models/model.joblib"]
        )
        validate_stage_artifacts(
            config, "build_prototypes", dependencies["build_prototypes"],
            ["prototypes/cluster_prototypes.parquet"],
        )
        model_path = "models/model.joblib"
        prototypes_path = "prototypes/cluster_prototypes.parquet"
    else:
        model_path, _, _ = _require_selected_candidate(config, selected_k)
        prototypes_path = _require_selected_candidate_prototypes(config, selected_k)
    test = storage.read_frame(config, "data/test_features.parquet")
    test_indicators = storage.read_frame(config, "data/test_indicators.parquet")
    prototypes_frame = storage.read_frame(config, prototypes_path)
    preprocessor = storage.read_object(config, "preprocessing/preprocessor.joblib")
    model = storage.read_object(config, model_path)
    prediction_fp = storage.stage_fingerprint(config, "predict_test_clusters")
    try:
        storage.require_stage(
            config,
            prediction_stage_key,
            prediction_fp,
            [prediction_path],
        )
        assignments = storage.read_frame(config, prediction_path)
    except FileNotFoundError:
        prediction_started = time.perf_counter()
        assigned = transformation.assign_test_clusters(test, config, preprocessor, model)
        prediction_seconds = time.perf_counter() - prediction_started
        assignments = assigned["assignments"]
        storage.write_frame(config, prediction_path, assignments)
        storage.record_stage(
            config,
            prediction_stage_key,
            prediction_fp,
            [prediction_path],
            {
                "split_dataset": dependencies["split_dataset"],
                "fit_preprocessor": dependencies["fit_preprocessor"],
                "train_model": dependencies["train_model"],
            },
            {"test_rows": len(test), "prediction_seconds": prediction_seconds},
        )
    if indicator_name not in test_indicators:
        raise ValueError(
            "Scenario {!r} is absent from saved test indicators; run generate_dataset "
            "and split_dataset with this scenario configured.".format(indicator_name)
        )
    replacement_started = time.perf_counter()
    transformed = transformation.transform_test(
        test,
        test_indicators[[config["dataset"]["id_column"], indicator_name]],
        prototypes_frame,
        assignments,
        config,
        indicator_name,
    )
    replacement_seconds = time.perf_counter() - replacement_started
    storage.write_frame(config, transformed_path, transformed["transformed"])
    storage.write_frame(config, original_path, transformed["original"])
    storage.record_stage(
        config,
        stage_key,
        fingerprint,
        [transformed_path, original_path, prediction_path],
        {
            "predict_test_clusters": prediction_fp,
            "build_prototypes": dependencies["build_prototypes"],
        },
        {
            "indicator": indicator_name,
            "test_rows": len(test),
            "replacement_seconds": replacement_seconds,
        },
    )
    return StageResult(
        "transform_test",
        {
            "transformed": transformed_path,
            "original": original_path,
            "test_assignments": prediction_path,
        },
        {
            "indicator": indicator_name,
            "test_rows": len(test),
            "n_replaced": int(
                transformed["transformed"]["gdpr"].eq(
                    config["transformation"]["replace_when"]
                ).sum()
            ),
        },
        config,
    )


def evaluate(config: LoadedConfig) -> StageResult:
    config = _require_config(config)
    if not config["evaluation"]["enabled"]:
        return StageResult("evaluate", {}, {"enabled": False}, config)
    transform_result = _run_transformation(
        config, config["transformation"]["selected_indicator"]
    )
    selected_k = storage.selected_candidate(config)
    if selected_k is None:
        model_path = "models/model.joblib"
        prototype_path = "prototypes/cluster_prototypes.parquet"
        train_record_key = "train_model"
        prediction_record_key = "predict_test_clusters"
        transform_record_key = "transform_test:{}".format(
            config["transformation"]["selected_indicator"]
        )
        metrics_path = "metrics/results.json"
        evaluation_stage_key = "evaluate"
    else:
        model_path, _, _ = _require_selected_candidate(config, selected_k)
        prototype_path = _require_selected_candidate_prototypes(config, selected_k)
        train_record_key = storage.candidate_stage_key("train_model", selected_k)
        prediction_record_key = "predict_test_clusters:k_{}".format(selected_k)
        transform_record_key = "transform_test:k_{}:{}".format(
            selected_k, config["transformation"]["selected_indicator"]
        )
        metrics_path = "metrics/constrained_kmeans/k_{}/results.json".format(
            selected_k
        )
        evaluation_stage_key = "evaluate:k_{}".format(selected_k)
    test = storage.read_frame(config, "data/test_features.parquet")
    test_indicators = storage.read_frame(config, "data/test_indicators.parquet")
    assignments = storage.read_frame(config, transform_result.artifacts["test_assignments"])
    train = storage.read_frame(config, "data/train_features.parquet")
    train_indicators = storage.read_frame(config, "data/train_indicators.parquet")
    model = storage.read_object(config, model_path)
    fitted_preprocessor = storage.read_object(config, "preprocessing/preprocessor.joblib")
    names = [feature["name"] for feature in config["features"]]
    train_inputs = fitted_preprocessor.transform(train[names])
    labels = np.asarray(model.labels_, dtype=int)
    unique, counts = np.unique(labels, return_counts=True)
    sizes = dict(zip(unique.tolist(), counts.tolist()))
    scenario_outputs = {}
    requested_rates = {}
    full_indicators = storage.read_frame(config, "data/nonconsent.parquet")
    prototype_frame = storage.read_frame(config, prototype_path)
    for column in test_indicators.columns:
        if column == config["dataset"]["id_column"]:
            continue
        requested_rates[column] = {
            "requested": int(column.rsplit("_", 1)[1]) / 100.0,
            "overall": float(
                full_indicators[column].mean()
            ),
            "train": float(train_indicators[column].mean()),
            "test": float(test_indicators[column].mean()),
        }
        transformed = transformation.transform_test(
            test,
            test_indicators[[config["dataset"]["id_column"], column]],
            prototype_frame,
            assignments,
            config,
            column,
        )
        scenario_outputs[column] = transformed["transformed"]
    manifest = storage.initialize_run(config)
    stage_records = manifest.get("stages", {})
    model_record = stage_records.get(train_record_key, {})
    prediction_record = stage_records.get(prediction_record_key, {})
    selected_record = stage_records.get(transform_record_key, {})
    report = evaluation.evaluate(
        train_inputs,
        labels,
        test,
        assignments,
        scenario_outputs,
        sizes,
        requested_rates,
        config,
        timings={
            "fitting": model_record.get("metadata", {}).get(
                "fit_seconds",
                model_record.get("metadata", {}).get("training_time_seconds"),
            ),
            "prediction": prediction_record.get("metadata", {}).get("prediction_seconds"),
            "replacement": selected_record.get("metadata", {}).get("replacement_seconds"),
        },
        artifact_sizes=storage.file_sizes(
            config,
            [
                "data/features.parquet",
                "data/train_features.parquet",
                "data/test_features.parquet",
                "preprocessing/preprocessor.joblib",
                model_path,
                prototype_path,
                transform_result.artifacts["test_assignments"],
                transform_result.artifacts["transformed"],
            ],
        ),
    )
    storage.write_json(config, metrics_path, report)
    fingerprint = storage.stage_fingerprint(config, "evaluate")
    dependencies =     {
        train_record_key: (
            storage.candidate_training_fingerprint(config, selected_k)
            if selected_k is not None
            else storage.stage_fingerprint(config, "train_model")
        ),
        "predict_test_clusters": storage.stage_fingerprint(config, "predict_test_clusters"),
        "build_prototypes": (
            storage.candidate_prototype_fingerprint(config, selected_k)
            if selected_k is not None
            else storage.stage_fingerprint(config, "build_prototypes")
        ),
    }
    storage.record_stage(
        config,
        evaluation_stage_key,
        fingerprint,
        [metrics_path],
        dependencies,
        {"scenario_count": len(scenario_outputs)},
    )
    return StageResult(
        "evaluate",
        {"metrics": metrics_path},
        {
            "scenario_count": len(scenario_outputs),
            "n_clusters": len(sizes),
            "minimum_training_cluster_size": min(sizes.values()),
        },
        config,
    )


def _require_config(config: LoadedConfig) -> LoadedConfig:
    if not isinstance(config, LoadedConfig):
        raise TypeError("Stage functions require a config returned by load_config()")
    return config


def _ids_fingerprint(values: pd.Series) -> str:
    import hashlib

    payload = ",".join(str(value) for value in values.tolist())
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
