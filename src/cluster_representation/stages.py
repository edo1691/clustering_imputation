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
from .robust_support import uses_candidate_artifacts
from .models import create_model
from .models.registry import training_constraints
from .randomness import generator
from .robust_support import (ROBUST_MODELS, ClusterPair, search_candidates,
                             candidate_from_record, candidate_feasibility)
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
    if fitted.imputer is None:
        raise RuntimeError("Fitted preprocessor is missing its training imputer")
    model_path = "preprocessing/preprocessor.joblib"
    schema_path = "preprocessing/schema.json"
    scales_path = "preprocessing/training_feature_scales.json"
    raw_training_values = train_features[names].to_numpy(
        dtype=float, na_value=np.nan
    )
    imputed_train = fitted.imputer.transform(raw_training_values)
    scales = np.std(imputed_train, axis=0, ddof=0)
    invalid_scales = [
        name for name, scale in zip(names, scales) if not np.isfinite(scale)
    ]
    if invalid_scales:
        raise ValueError(
            "Non-finite training standard deviation for feature(s): {}".format(
                ", ".join(invalid_scales)
            )
        )
    scale_metadata = {
        "feature_order": names,
        "feature_scales": {
            name: float(scale) for name, scale in zip(names, scales)
        },
        "ddof": 0,
        "imputation_policy": config["preprocessing"]["imputation"],
        "training_row_count": len(train_features),
        "training_id_fingerprint": _ids_fingerprint(train_features[id_column]),
        "constant_features": [
            name for name, scale in zip(names, scales) if scale == 0
        ],
    }
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
    storage.write_json(config, scales_path, scale_metadata)
    fingerprint = storage.stage_fingerprint(config, "fit_preprocessor")
    storage.record_stage(
        config,
        "fit_preprocessor",
        fingerprint,
        [model_path, schema_path, scales_path],
        {"split_dataset": split_fp},
        {
            "feature_order": names,
            "training_id_count": len(train_features),
            "training_feature_scales": scales_path,
            "scale_ddof": 0,
            "imputation_policy": config["preprocessing"]["imputation"],
        },
    )
    return StageResult(
        "fit_preprocessor",
        {
            "preprocessor": model_path,
            "schema": schema_path,
            "training_feature_scales": scales_path,
        },
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
    if uses_candidate_artifacts(config):
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
    artifacts = {"model": model_path, "metadata": metadata_path}
    training_artifacts = [model_path, metadata_path]
    if config.model_config["name"] in {"spectral_coclustering", "robust_coclustering", "robust_constrained_coclustering"}:
        preprocessing_path = "models/model_preprocessor.joblib"
        column_labels_path = "models/column_labels.parquet"
        storage.write_object(config, preprocessing_path, model.preprocessing_)
        storage.write_frame(
            config,
            column_labels_path,
            pd.DataFrame(
                {
                    "feature_index": np.arange(len(names)),
                    "feature": names,
                    "column_cluster": model.column_labels_,
                }
            ),
        )
        artifacts.update(
            {
                "model_preprocessor": preprocessing_path,
                "column_labels": column_labels_path,
            }
        )
        training_artifacts.extend([preprocessing_path, column_labels_path])
    storage.write_object(config, model_path, model)
    training_labels = np.asarray(model.labels_, dtype=int)
    metadata = model.metadata()
    metadata["training_id_fingerprint"] = _ids_fingerprint(
        train_features[config["dataset"]["id_column"]]
    )
    metadata["feature_order"] = names
    metadata["fit_seconds"] = fit_seconds
    metadata["training_time_seconds"] = fit_seconds
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
        training_artifacts,
        {"fit_preprocessor": preprocessor_fp},
        {
            "training_rows": len(train_features),
            "model": metadata["name"],
            "fit_seconds": fit_seconds,
        },
    )
    return StageResult(
        "train_model",
        artifacts,
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
    model_name = config.model_config["name"]
    constraints_applicable = model_name in {"constrained_kmeans", "robust_coclustering", "robust_constrained_coclustering"}
    constraints = training_constraints(config.model_config)
    n_train = len(train_features)
    id_column = config["dataset"]["id_column"]
    seed = config.model_config["parameters"].get("seed", config.model_config["parameters"].get("random_state"))
    if seed is None:
        seed = config["dataset"]["seed"]
    sample_size = min(n_train, config["evaluation"]["silhouette_sample_size"])
    sample_rng = generator(seed, "cluster-search:silhouette")
    silhouette_indices = np.sort(
        sample_rng.choice(n_train, size=sample_size, replace=False)
    )
    rows = []
    candidate_artifacts: Dict[str, str] = {}
    minimum_size = (
        training_constraints(config.model_config)["min_cluster_size"]
        if constraints_applicable
        else None
    )

    for candidate in search_candidates(config):
        base = "models/{}/k_{}".format(model_name, candidate)
        paths = {
            "model": base + "/model.joblib",
            "assignments": base + "/train_cluster_assignments.parquet",
            "metadata": base + "/metadata.json",
        }
        if model_name in {"spectral_coclustering", "robust_coclustering", "robust_constrained_coclustering"}:
            paths["column_labels"] = base + "/column_labels.parquet"
            paths["model_preprocessor"] = base + "/model_preprocessor.joblib"
        stage_key = storage.candidate_stage_key(
            "train_model", candidate, model_name
        )
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
            feasibility_reason = None
            if isinstance(candidate, ClusterPair):
                fit_parameters = dict(config.model_config["parameters"], g=candidate.g, m=candidate.m)
                feasibility_reason = candidate_feasibility(fit_parameters, train_inputs)
            if feasibility_reason or (constraints_applicable and (candidate.g if isinstance(candidate, ClusterPair) else candidate) * minimum_size > n_train):
                candidate_result = _candidate_result(
                    candidate,
                    status="skipped_infeasible",
                    failure_reason=feasibility_reason or (
                        "{} * {} exceeds n_train={}".format(
                            candidate, minimum_size, n_train
                        )
                    ),
                    constraint_applicability="applicable",
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
                    "name": model_name,
                    "parameters": dict(config.model_config["parameters"]),
                }
                if model_name == "constrained_kmeans":
                    candidate_model_config["constraints"] = constraints
                if isinstance(candidate, ClusterPair):
                    candidate_model_config["parameters"].update(g=candidate.g, m=candidate.m)
                else:
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
                        constraint_applicability=(
                            "applicable"
                            if constraints_applicable
                            else "not applicable"
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
                    training_seconds = time.perf_counter() - started
                    labels = np.asarray(model.labels_, dtype=int)
                    requested_rows = candidate.g if isinstance(candidate, ClusterPair) else candidate
                    counts = np.bincount(labels, minlength=requested_rows + 1)[1:]
                    if len(counts) != requested_rows:
                        raise RuntimeError(
                            "Candidate {} returned an unexpected number of clusters".format(
                                candidate
                            )
                        )
                    violations = (
                        np.flatnonzero((counts < minimum_size) | (counts > (constraints.get("max_cluster_size") or n_train))) + 1
                        if constraints_applicable
                        else np.array([], dtype=int)
                    )
                    metrics = evaluation.evaluate_training_candidate(
                        train_inputs, labels, silhouette_indices
                    )
                    model_metadata = model.metadata()
                    candidate_result = _candidate_result(
                        candidate,
                        status="success",
                        failure_reason=None,
                        training_time_seconds=training_seconds,
                        actual_clusters=int(len(np.unique(labels))),
                        actual_column_clusters=int(len(np.unique(model.column_labels_))) if isinstance(candidate, ClusterPair) else None,
                        model_metadata=json.dumps(model_metadata, sort_keys=True),
                        constraint_violations=int(len(violations)),
                        constraint_violation_clusters=violations.tolist(),
                        constraint_applicability=(
                            "applicable"
                            if constraints_applicable
                            else "not applicable"
                        ),
                        training_id_fingerprint=_ids_fingerprint(
                            train_features[id_column]
                        ),
                        feature_order=[feature["name"] for feature in config["features"]],
                        model_parameters=json.dumps(
                            candidate_model_config["parameters"], sort_keys=True
                        ),
                        constraints=(
                            json.dumps(constraints, sort_keys=True)
                            if constraints_applicable
                            else None
                        ),
                        assignment_strategy=model_metadata.get(
                            "assignment_strategy", "model.predict"
                        ),
                        model_preprocessing=model_metadata.get("preprocessing"),
                        metric_feature_space=(
                            "shared fitted preprocessing output: imputation=median, "
                            "scaling={}".format(config["preprocessing"]["scaling"])
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
                    if model_name in {"spectral_coclustering", "robust_coclustering", "robust_constrained_coclustering"}:
                        feature_names = [
                            feature["name"] for feature in config["features"]
                        ]
                        storage.write_frame(
                            config,
                            paths["column_labels"],
                            pd.DataFrame(
                                {
                                    "feature_index": np.arange(len(feature_names)),
                                    "feature": feature_names,
                                    "column_cluster": model.column_labels_,
                                }
                            ),
                        )
                        storage.write_object(
                            config,
                            paths["model_preprocessor"],
                            model.preprocessing_,
                        )
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
            for key in ("column_labels", "model_preprocessor"):
                if key in paths:
                    candidate_artifacts["{}_k_{}".format(key, candidate)] = paths[key]

    comparison = pd.DataFrame(rows)
    search_fingerprint = storage.cluster_search_fingerprint(config)
    comparison_path = "models/{}/search_{}/comparison.parquet".format(
        model_name, search_fingerprint[:12]
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
                storage.candidate_stage_key(
                    "train_model", candidate, model_name
                ):
                    storage.candidate_training_fingerprint(config, candidate)
                for candidate in search_candidates(config)
            },
            {"candidate_count": len(search["candidates"])},
        )
    else:
        comparison = storage.read_frame(config, comparison_path)
    candidate_artifacts["comparison"] = comparison_path
    selection_fingerprint = storage.selection_fingerprint(config)
    selection_stage_key = storage.selection_stage_key(config)
    selection_path = storage.selection_artifact_path(config)
    try:
        storage.require_stage(
            config,
            selection_stage_key,
            selection_fingerprint,
            [selection_path],
        )
        selection = storage.read_json(config, selection_path)
    except FileNotFoundError:
        selection_candidates = []
        for candidate in comparison.to_dict(orient="records"):
            candidate = dict(candidate)
            parameters = dict(config.model_config["parameters"])
            if model_name in ROBUST_MODELS:
                parameters.update(g=int(candidate["n_clusters"]), m=int(candidate["requested_column_clusters"]))
            else:
                parameters["n_clusters"] = int(candidate["n_clusters"])
            if not isinstance(candidate.get("model_parameters"), str):
                candidate["model_parameters"] = json.dumps(
                    parameters, sort_keys=True
                )
            if constraints_applicable and not isinstance(candidate.get("constraints"), str):
                candidate["constraints"] = json.dumps(
                    constraints, sort_keys=True
                )
            selection_candidates.append(candidate)
        selection = evaluation.resolve_cluster_selection(
            model_name,
            selection_candidates,
            search["selected_n_clusters"],
            constraints,
            constraints_applicable=constraints_applicable,
        )
        storage.write_json(config, selection_path, selection)
        storage.record_stage(
            config,
            selection_stage_key,
            selection_fingerprint,
            [selection_path],
            {search_stage_key: search_fingerprint},
            {
                "selection_mode": selection["selection_mode"],
                "selected_n_clusters": selection["selected_n_clusters"],
                "selection_rule": selection["selection_rule"],
                "excluded_candidate_count": len(selection["excluded_candidates"]),
            },
        )
    candidate_artifacts["selection"] = selection_path
    selected_k = storage.selected_candidate(config)
    selected_mask = comparison["n_clusters"].eq(selected_k.g if isinstance(selected_k, ClusterPair) else selected_k)
    if isinstance(selected_k, ClusterPair):
        selected_mask &= comparison["requested_column_clusters"].eq(selected_k.m)
    selected_row = comparison.loc[selected_mask].iloc[0]
    if selected_row["status"] != "success":
        raise ValueError(
            "Selected cluster count {} was not successfully trained ({}: {}).".format(
                selected_k,
                selected_row["status"],
                selected_row.get("failure_reason") or "no failure details",
            )
        )
    return StageResult(
        "train_model",
        candidate_artifacts,
        {
            "model": model_name,
            "fixed_m": config.model_config["parameters"].get("m"),
            "training_rows": n_train,
            "candidate_count": len(search["candidates"]),
            "selection_mode": selection["selection_mode"],
            "selected_n_clusters": selected_k.g if isinstance(selected_k, ClusterPair) else selected_k,
            "selected_pair": selection.get("selected_pair"),
            "selection_rule": selection["selection_rule"],
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
        "n_clusters": n_clusters.g if isinstance(n_clusters, ClusterPair) else n_clusters,
        "requested_row_clusters": n_clusters.g if isinstance(n_clusters, ClusterPair) else n_clusters,
        "requested_column_clusters": n_clusters.m if isinstance(n_clusters, ClusterPair) else None,
        "actual_clusters": None,
        "actual_column_clusters": None,
        "candidate_id": str(n_clusters),
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


def _candidate_training_paths(
    n_clusters: int, model_name: str = "constrained_kmeans"
) -> Tuple[str, str, str]:
    base = "models/{}/k_{}".format(model_name, n_clusters)
    return (
        base + "/model.joblib",
        base + "/train_cluster_assignments.parquet",
        base + "/metadata.json",
    )


def _candidate_prototype_path(
    n_clusters: int, model_name: str = "constrained_kmeans"
) -> str:
    return "models/{}/k_{}/cluster_prototypes.parquet".format(
        model_name, n_clusters
    )


def _cluster_search_comparison_path(config: LoadedConfig) -> str:
    fingerprint = storage.cluster_search_fingerprint(config)
    return "models/{}/search_{}/comparison.parquet".format(
        config.model_config["name"],
        fingerprint[:12]
    )


def _cluster_search_stage_key(config: LoadedConfig) -> str:
    fingerprint = storage.cluster_search_fingerprint(config)
    return "train_model:search:{}".format(fingerprint[:12])


def _require_selected_candidate(config: LoadedConfig, n_clusters: int) -> tuple:
    model_name = config.model_config["name"]
    model_path, assignments_path, metadata_path = _candidate_training_paths(
        n_clusters, model_name
    )
    required = [model_path, assignments_path, metadata_path]
    if model_name in {"spectral_coclustering", "robust_coclustering", "robust_constrained_coclustering"}:
        required.extend(
            [
                "models/{}/k_{}/column_labels.parquet".format(model_name, n_clusters),
                "models/{}/k_{}/model_preprocessor.joblib".format(
                    model_name, n_clusters
                ),
            ]
        )
    key = storage.candidate_stage_key("train_model", n_clusters, model_name)
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
        config, key, fingerprint, required
    )
    return model_path, assignments_path, metadata_path


def _require_selected_candidate_prototypes(
    config: LoadedConfig, n_clusters: int
) -> str:
    model_name = config.model_config["name"]
    path = _candidate_prototype_path(n_clusters, model_name)
    key = storage.candidate_stage_key("build_prototypes", n_clusters, model_name)
    fingerprint = storage.candidate_prototype_fingerprint(config, n_clusters)
    storage.require_stage(config, key, fingerprint, [path])
    return path


def build_prototypes(config: LoadedConfig) -> StageResult:
    config = _require_config(config)
    if uses_candidate_artifacts(config):
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
    minimum = training_constraints(config.model_config).get("min_cluster_size", 1)
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
    model_name = config.model_config["name"]
    minimum_size = (
        training_constraints(config.model_config)["min_cluster_size"]
        if model_name in {"constrained_kmeans", "robust_coclustering", "robust_constrained_coclustering"}
        else 1
    )
    artifacts: Dict[str, str] = {"comparison": comparison_path}
    candidate_summaries = []

    for candidate_row in comparison.to_dict(orient="records"):
        candidate = candidate_from_record(config, candidate_row)
        if candidate_row["status"] != "success":
            continue
        model_path, assignment_path, metadata_path = _candidate_training_paths(
            candidate, model_name
        )
        candidate_train_key = storage.candidate_stage_key(
            "train_model", candidate, model_name
        )
        candidate_required = [model_path, assignment_path, metadata_path]
        if model_name in {"spectral_coclustering", "robust_coclustering", "robust_constrained_coclustering"}:
            candidate_required.extend(
                [
                    "models/{}/k_{}/column_labels.parquet".format(
                        model_name, candidate
                    ),
                    "models/{}/k_{}/model_preprocessor.joblib".format(
                        model_name, candidate
                    ),
                ]
            )
        candidate_train_fingerprint = storage.candidate_training_fingerprint(
            config, candidate
        )
        training_record = storage.require_stage(
            config,
            candidate_train_key,
            candidate_train_fingerprint,
            candidate_required,
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
        prototype_path = _candidate_prototype_path(candidate, model_name)
        prototype_key = storage.candidate_stage_key(
            "build_prototypes", candidate, model_name
        )
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
                "n_clusters": candidate.g if isinstance(candidate, ClusterPair) else candidate,
                "column_clusters": candidate.m if isinstance(candidate, ClusterPair) else None,
                "minimum_support": int(prototype_frame["n_clients"].min()),
            }
        )

    selected = storage.selected_candidate(config)
    if selected is None:
        raise RuntimeError("Cluster search did not resolve a selected candidate")
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
    artifacts["assignments"] = _candidate_training_paths(selected, model_name)[1]
    return StageResult(
        "build_prototypes",
        artifacts,
        {
            "candidate_prototypes": candidate_summaries,
            "selected_n_clusters": selected.g if isinstance(selected, ClusterPair) else selected,
            "selected_pair": {"g": selected.g, "m": selected.m} if isinstance(selected, ClusterPair) else None,
            "selection_mode": search["selected_n_clusters"]
            if search["selected_n_clusters"] == "auto"
            else "fixed",
        },
        config,
    )


def transform_test(config: LoadedConfig) -> StageResult:
    config = _require_config(config)
    selected_k = storage.selected_candidate(config)
    fingerprint = storage.stage_fingerprint(config, "transform_test")
    model_name = config.model_config["name"]
    if selected_k is None:
        prediction_stage_key = "predict_test_clusters"
        prediction_path = "predictions/test_cluster_assignments.parquet"
    else:
        model_stage_id = (
            "k_{}".format(selected_k)
            if model_name == "constrained_kmeans"
            else "{}:k_{}".format(model_name, selected_k)
        )
        prediction_stage_key = "predict_test_clusters:{}".format(model_stage_id)
        prediction_path = (
            "predictions/{}/k_{}/test_cluster_assignments.parquet".format(
                model_name, selected_k
            )
        )
    stage_key = "transform_test:{}{}".format(
        model_name,
        ":k_{}".format(selected_k) if selected_k is not None else "",
    )
    transformed_prefix = "models/{}/transformed".format(model_name)
    if selected_k is not None:
        transformed_prefix = "models/{}/k_{}/transformed".format(
            model_name, selected_k
        )
    manifest_path = storage.artifact_path(config, storage.MANIFEST_NAME)
    test_indicators = storage.read_frame(config, "data/test_indicators.parquet")
    scenario_names = nonconsent.scenario_columns(config)
    missing_scenarios = [
        name for name in scenario_names if name not in test_indicators.columns
    ]
    if missing_scenarios:
        raise ValueError(
            "Configured non-consent indicator(s) are missing from saved test "
            "indicators: {}".format(", ".join(missing_scenarios))
        )
    scenario_artifacts = {}
    scenario_paths = {}
    for indicator_name in scenario_names:
        scenario_dir = "{}/{}".format(transformed_prefix, indicator_name)
        transformed_path = scenario_dir + "/transformed_test.parquet"
        original_path = scenario_dir + "/original_test.parquet"
        scenario_paths[indicator_name] = (transformed_path, original_path)
        scenario_artifacts["transformed_{}".format(indicator_name)] = transformed_path
        scenario_artifacts["original_{}".format(indicator_name)] = original_path
    required_artifacts = [
        path
        for paths in scenario_paths.values()
        for path in paths
    ] + [prediction_path]
    if manifest_path.exists():
        manifest = storage.initialize_run(config)
        existing_record = manifest.get("stages", {}).get(stage_key)
        if existing_record and existing_record.get("fingerprint") == fingerprint:
            validate_stage_artifacts(
                config,
                stage_key,
                fingerprint,
                required_artifacts,
            )
            replaced_counts = {
                name: int(
                    storage.read_frame(config, paths[0])["gdpr"]
                    .eq(config["transformation"]["replace_when"])
                    .sum()
                )
                for name, paths in scenario_paths.items()
            }
            return StageResult(
                "transform_test",
                {
                    **scenario_artifacts,
                    "test_assignments": prediction_path,
                },
                {
                    "scenarios": scenario_names,
                    "test_rows": len(
                        storage.read_frame(config, "data/test_features.parquet")
                    ),
                    "n_replaced_by_scenario": replaced_counts,
                },
                config,
            )
        if existing_record and not config["output"]["overwrite"]:
            raise ValueError(
                "Saved transformations are incompatible with the configured NC "
                "scenarios. Set output.overwrite=true or use a new run_name."
            )
    dependencies = {
        "split_dataset": storage.stage_fingerprint(config, "split_dataset"),
        "fit_preprocessor": storage.stage_fingerprint(config, "fit_preprocessor"),
        "train_model": storage.stage_fingerprint(config, "train_model"),
        "build_prototypes": storage.stage_fingerprint(config, "build_prototypes"),
    }
    validate_stage_artifacts(
        config,
        "split_dataset",
        dependencies["split_dataset"],
        ["data/test_features.parquet"],
    )
    validate_stage_artifacts(
        config,
        "fit_preprocessor",
        dependencies["fit_preprocessor"],
        [
            "preprocessing/preprocessor.joblib",
            "preprocessing/training_feature_scales.json",
        ],
    )
    if selected_k is None:
        validate_stage_artifacts(
            config, "train_model", dependencies["train_model"], ["models/model.joblib"]
        )
        validate_stage_artifacts(
            config,
            "build_prototypes",
            dependencies["build_prototypes"],
            ["prototypes/cluster_prototypes.parquet"],
        )
        model_path = "models/model.joblib"
        prototypes_path = "prototypes/cluster_prototypes.parquet"
    else:
        model_path, _, _ = _require_selected_candidate(config, selected_k)
        prototypes_path = _require_selected_candidate_prototypes(config, selected_k)
    test = storage.read_frame(config, "data/test_features.parquet")
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
        assigned = transformation.assign_test_clusters(
            test, config, preprocessor, model
        )
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
            {
                "test_rows": len(test),
                "prediction_seconds": prediction_seconds,
                "assignment_strategy": getattr(
                    model,
                    "assignment_strategy",
                    "model.predict",
                ),
            },
        )
    replacement_seconds = 0.0
    replaced_counts = {}
    for indicator_name in scenario_names:
        transformed_path, original_path = scenario_paths[indicator_name]
        replacement_started = time.perf_counter()
        transformed = transformation.transform_test(
            test,
            test_indicators[
                [config["dataset"]["id_column"], indicator_name]
            ],
            prototypes_frame,
            assignments,
            config,
            indicator_name,
        )
        replacement_seconds += time.perf_counter() - replacement_started
        storage.write_frame(config, transformed_path, transformed["transformed"])
        storage.write_frame(config, original_path, transformed["original"])
        replaced_counts[indicator_name] = int(
            transformed["transformed"]["gdpr"]
            .eq(config["transformation"]["replace_when"])
            .sum()
        )
    storage.record_stage(
        config,
        stage_key,
        fingerprint,
        required_artifacts,
        {
            "predict_test_clusters": prediction_fp,
            "build_prototypes": dependencies["build_prototypes"],
        },
        {
            "scenarios": scenario_names,
            "test_rows": len(test),
            "replacement_seconds_total": replacement_seconds,
            "n_replaced_by_scenario": replaced_counts,
        },
    )
    return StageResult(
        "transform_test",
        {
            **scenario_artifacts,
            "test_assignments": prediction_path,
        },
        {
            "scenarios": scenario_names,
            "test_rows": len(test),
            "n_replaced_by_scenario": replaced_counts,
        },
        config,
    )


def evaluate(config: LoadedConfig) -> StageResult:
    config = _require_config(config)
    if not config["evaluation"]["enabled"]:
        return StageResult("evaluate", {}, {"enabled": False}, config)
    transform_result = transform_test(config)
    selected_k = storage.selected_candidate(config)
    model_name = config.model_config["name"]
    if selected_k is None:
        model_path = "models/model.joblib"
        prototype_path = "prototypes/cluster_prototypes.parquet"
        train_record_key = "train_model"
        prediction_record_key = "predict_test_clusters"
        metrics_path = "metrics/results.json"
    else:
        model_path, _, _ = _require_selected_candidate(config, selected_k)
        prototype_path = _require_selected_candidate_prototypes(config, selected_k)
        train_record_key = storage.candidate_stage_key(
            "train_model", selected_k, model_name
        )
        model_stage_id = (
            "k_{}".format(selected_k)
            if model_name == "constrained_kmeans"
            else "{}:k_{}".format(model_name, selected_k)
        )
        prediction_record_key = "predict_test_clusters:{}".format(model_stage_id)
        metrics_path = "metrics/{}/k_{}/results.json".format(
            model_name, selected_k
        )
    transform_record_key = "transform_test:{}{}".format(
        model_name,
        ":k_{}".format(selected_k) if selected_k is not None else "",
    )
    evaluation_stage_key = "evaluate:{}{}".format(
        model_name,
        ":k_{}".format(selected_k) if selected_k is not None else "",
    )
    test = storage.read_frame(config, "data/test_features.parquet")
    test_indicators = storage.read_frame(config, "data/test_indicators.parquet")
    assignments = storage.read_frame(
        config, transform_result.artifacts["test_assignments"]
    )
    train = storage.read_frame(config, "data/train_features.parquet")
    train_indicators = storage.read_frame(config, "data/train_indicators.parquet")
    model = storage.read_object(config, model_path)
    fitted_preprocessor = storage.read_object(config, "preprocessing/preprocessor.joblib")
    names = [feature["name"] for feature in config["features"]]
    scale_metadata = storage.read_json(
        config, "preprocessing/training_feature_scales.json"
    )
    if scale_metadata.get("feature_order") != names:
        raise ValueError(
            "Saved training feature scales have an incompatible feature order"
        )
    if scale_metadata.get("ddof") != 0:
        raise ValueError("Saved training feature scales must use ddof=0")
    if scale_metadata.get("imputation_policy") != config["preprocessing"]["imputation"]:
        raise ValueError(
            "Saved training feature scales use an incompatible imputation policy"
        )
    if scale_metadata.get("training_id_fingerprint") != _ids_fingerprint(
        train[config["dataset"]["id_column"]]
    ):
        raise ValueError("Saved training feature scales use a different training split")
    training_feature_stds = {
        name: scale_metadata["feature_scales"][name] for name in names
    }
    train_inputs = fitted_preprocessor.transform(train[names])
    labels = np.asarray(model.labels_, dtype=int)
    unique, counts = np.unique(labels, return_counts=True)
    sizes = dict(zip(unique.tolist(), counts.tolist()))
    scenario_outputs = {}
    requested_rates = {}
    full_indicators = storage.read_frame(config, "data/nonconsent.parquet")
    scenario_names = nonconsent.scenario_columns(config)
    missing_scenarios = [
        name for name in scenario_names if name not in test_indicators.columns
    ]
    if missing_scenarios:
        raise ValueError(
            "Configured non-consent indicator(s) are missing from saved test "
            "indicators: {}".format(", ".join(missing_scenarios))
        )
    transformed_artifacts = []
    id_column = config["dataset"]["id_column"]
    for column in scenario_names:
        transformed_path = transform_result.artifacts[
            "transformed_{}".format(column)
        ]
        original_path = transform_result.artifacts["original_{}".format(column)]
        transformed = storage.read_frame(config, transformed_path)
        original = storage.read_frame(config, original_path)
        if original[id_column].tolist() != test[id_column].tolist():
            raise ValueError(
                "Saved original rows for scenario {!r} do not align with test rows".format(
                    column
                )
            )
        if transformed[id_column].tolist() != test[id_column].tolist():
            raise ValueError(
                "Saved transformed rows for scenario {!r} do not align with test rows".format(
                    column
                )
            )
        scenario_outputs[column] = transformed
        transformed_artifacts.extend([transformed_path, original_path])
        percent = int(column.rsplit("_", 1)[1])
        requested_rates[column] = {
            "requested": percent / 100.0,
            "overall": float(
                full_indicators[column].mean()
            ),
            "train": float(train_indicators[column].mean()),
            "test": float(test_indicators[column].mean()),
        }
    manifest = storage.initialize_run(config)
    stage_records = manifest.get("stages", {})
    model_record = stage_records.get(train_record_key, {})
    prediction_record = stage_records.get(prediction_record_key, {})
    selected_record = stage_records.get(transform_record_key, {})
    artifact_paths = [
        "data/features.parquet",
        "data/train_features.parquet",
        "data/test_features.parquet",
        "preprocessing/preprocessor.joblib",
        "preprocessing/training_feature_scales.json",
        model_path,
        prototype_path,
        transform_result.artifacts["test_assignments"],
        *transformed_artifacts,
    ]
    if model_name in {"spectral_coclustering", "robust_coclustering", "robust_constrained_coclustering"}:
        if selected_k is None:
            artifact_paths.extend(
                ["models/model_preprocessor.joblib", "models/column_labels.parquet"]
            )
        else:
            base = "models/{}/k_{}".format(model_name, selected_k)
            artifact_paths.extend(
                [
                    base + "/model_preprocessor.joblib",
                    base + "/column_labels.parquet",
                ]
            )
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
            "replacement": selected_record.get("metadata", {}).get(
                "replacement_seconds_total"
            ),
        },
        artifact_sizes=storage.file_sizes(config, artifact_paths),
        training_feature_stds=training_feature_stds,
    )
    saved_model_metadata = model.metadata()
    report["constraint_applicability"] = saved_model_metadata.get(
        "constraint_applicability",
        "applicable" if model_name == "constrained_kmeans" else "not applicable",
    )
    if model_name in ROBUST_MODELS:
        report["model_metadata"] = saved_model_metadata
        report["requested_row_clusters"] = model.parameters["g"]
        report["requested_column_clusters"] = model.parameters["m"]
        report["actual_column_clusters"] = saved_model_metadata["n_column_clusters"]
    report["assignment_strategy"] = saved_model_metadata.get(
        "assignment_strategy", "model.predict"
    )
    report["model_preprocessing"] = saved_model_metadata.get("preprocessing")
    storage.write_json(config, metrics_path, report)
    comparison_markdown_path = reporting.export_overall_comparison(config, report)
    feature_evaluation_path = reporting.export_feature_evaluation(config, report)
    cluster_selection_path = reporting.export_cluster_selection_comparison(
        config, report
    )
    fingerprint = storage.stage_fingerprint(config, "evaluate")
    dependencies = {
        train_record_key: (
            storage.candidate_training_fingerprint(config, selected_k)
            if selected_k is not None
            else storage.stage_fingerprint(config, "train_model")
        ),
        "predict_test_clusters": storage.stage_fingerprint(config, "predict_test_clusters"),
        transform_record_key: storage.stage_fingerprint(config, "transform_test"),
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
        [
            metrics_path,
            comparison_markdown_path,
            feature_evaluation_path,
            cluster_selection_path,
        ],
        dependencies,
        {"scenario_count": len(scenario_outputs)},
    )
    return StageResult(
        "evaluate",
        {
            "metrics": metrics_path,
            "overall_comparison": comparison_markdown_path,
            "feature_evaluation": feature_evaluation_path,
            "cluster_selection_comparison": cluster_selection_path,
        },
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
