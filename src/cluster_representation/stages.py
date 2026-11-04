"""Public, artifact-backed workflow stages shared by CLI and notebooks."""

from dataclasses import dataclass
import time
from typing import Any, Dict, Mapping, Optional

import numpy as np
import pandas as pd

from . import evaluation
from . import generation, nonconsent, preprocessing, prototypes, reporting, splitting
from . import storage, transformation
from .config import LoadedConfig
from .models import create_model
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
    model = create_model(config.model_config)
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


def build_prototypes(config: LoadedConfig) -> StageResult:
    config = _require_config(config)
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


def transform_test(config: LoadedConfig) -> StageResult:
    config = _require_config(config)
    transformed = _run_transformation(config, config["transformation"]["selected_indicator"])
    return transformed


def _run_transformation(config: LoadedConfig, indicator_name: str) -> StageResult:
    fingerprint = storage.stage_fingerprint(config, "transform_test")
    stage_key = "transform_test:{}".format(indicator_name)
    transformed_path = "transformed/{}/transformed_test.parquet".format(indicator_name)
    original_path = "transformed/{}/original_test.parquet".format(indicator_name)
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
                    "test_assignments": "predictions/test_cluster_assignments.parquet",
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
    validate_stage_artifacts(
        config, "train_model", dependencies["train_model"], ["models/model.joblib"]
    )
    validate_stage_artifacts(
        config, "build_prototypes", dependencies["build_prototypes"],
        ["prototypes/cluster_prototypes.parquet"],
    )
    test = storage.read_frame(config, "data/test_features.parquet")
    test_indicators = storage.read_frame(config, "data/test_indicators.parquet")
    prototypes_frame = storage.read_frame(config, "prototypes/cluster_prototypes.parquet")
    preprocessor = storage.read_object(config, "preprocessing/preprocessor.joblib")
    model = storage.read_object(config, "models/model.joblib")
    prediction_fp = storage.stage_fingerprint(config, "predict_test_clusters")
    prediction_path = "predictions/test_cluster_assignments.parquet"
    try:
        storage.require_stage(
            config,
            "predict_test_clusters",
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
            "predict_test_clusters",
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
    scenario_dir = "transformed/{}".format(indicator_name)
    transformed_path = scenario_dir + "/transformed_test.parquet"
    original_path = scenario_dir + "/original_test.parquet"
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
    _run_transformation(config, config["transformation"]["selected_indicator"])
    test = storage.read_frame(config, "data/test_features.parquet")
    test_indicators = storage.read_frame(config, "data/test_indicators.parquet")
    assignments = storage.read_frame(
        config, "predictions/test_cluster_assignments.parquet"
    )
    train = storage.read_frame(config, "data/train_features.parquet")
    train_indicators = storage.read_frame(config, "data/train_indicators.parquet")
    model = storage.read_object(config, "models/model.joblib")
    fitted_preprocessor = storage.read_object(config, "preprocessing/preprocessor.joblib")
    names = [feature["name"] for feature in config["features"]]
    train_inputs = fitted_preprocessor.transform(train[names])
    labels = np.asarray(model.labels_, dtype=int)
    unique, counts = np.unique(labels, return_counts=True)
    sizes = dict(zip(unique.tolist(), counts.tolist()))
    scenario_outputs = {}
    requested_rates = {}
    full_indicators = storage.read_frame(config, "data/nonconsent.parquet")
    prototype_frame = storage.read_frame(config, "prototypes/cluster_prototypes.parquet")
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
    model_record = stage_records.get("train_model", {})
    prediction_record = stage_records.get("predict_test_clusters", {})
    selected_record = stage_records.get(
        "transform_test:{}".format(config["transformation"]["selected_indicator"]), {}
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
            "fitting": model_record.get("metadata", {}).get("fit_seconds"),
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
                "models/model.joblib",
                "prototypes/train_cluster_assignments.parquet",
                "prototypes/cluster_prototypes.parquet",
                "predictions/test_cluster_assignments.parquet",
                "transformed/{}/transformed_test.parquet".format(
                    config["transformation"]["selected_indicator"]
                ),
            ],
        ),
    )
    metrics_path = "metrics/results.json"
    storage.write_json(config, metrics_path, report)
    fingerprint = storage.stage_fingerprint(config, "evaluate")
    dependencies = {
        "train_model": storage.stage_fingerprint(config, "train_model"),
        "predict_test_clusters": storage.stage_fingerprint(config, "predict_test_clusters"),
        "build_prototypes": storage.stage_fingerprint(config, "build_prototypes"),
    }
    storage.record_stage(
        config,
        "evaluate",
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
