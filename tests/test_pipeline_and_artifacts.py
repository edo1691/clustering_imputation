import copy
import json

import nbformat
import numpy as np
import pandas as pd
import pytest
import yaml

from cluster_representation import load_config
from cluster_representation import pipeline, stages, storage
from cluster_representation.evaluation import evaluate


def test_end_to_end_pipeline_writes_reusable_artifacts(small_config):
    results = pipeline.run(small_config)
    assert [result.stage for result in results] == small_config["pipeline"]["steps"]
    root = small_config.artifact_dir
    assert (root / "manifest.json").is_file()
    assert (root / "resolved_config.yml").is_file()
    assert (root / "environment.json").is_file()
    assert (root / "prototypes/train_cluster_assignments.parquet").is_file()
    assert (root / "prototypes/cluster_prototypes.parquet").is_file()
    assert (root / "predictions/test_cluster_assignments.parquet").is_file()
    assert (root / "metrics/results.json").is_file()
    transformed = pd.read_parquet(
        root / "transformed/gdpr_20/transformed_test.parquet"
    )
    originals = pd.read_parquet(root / "transformed/gdpr_20/original_test.parquet")
    assert transformed["id_row"].tolist() == originals["id_row"].tolist()
    assert len(transformed) == 50

    metrics = json.loads((root / "metrics/results.json").read_text(encoding="utf-8"))
    assert len(metrics["scenarios"]) == 6
    assert metrics["scenarios"]["gdpr_0"]["replaced_rows_mae"] is None
    assert metrics["scenarios"]["gdpr_0"]["replaced_rows_mae_status"] == "unavailable_no_replaced_rows"


def test_changing_selected_scenario_reuses_model_and_cluster_assignments(small_config):
    pipeline.run(small_config)
    root = small_config.artifact_dir
    assignments_before = pd.read_parquet(
        root / "predictions/test_cluster_assignments.parquet"
    )
    model_path = root / "models/model.joblib"
    model_timestamp = model_path.stat().st_mtime_ns
    alternate = copy.deepcopy(small_config)
    alternate["transformation"]["selected_indicator"] = "gdpr_80"

    result = stages.transform_test(alternate)

    assignments_after = pd.read_parquet(
        root / "predictions/test_cluster_assignments.parquet"
    )
    pd.testing.assert_frame_equal(assignments_before, assignments_after)
    assert model_path.stat().st_mtime_ns == model_timestamp
    assert result.summary["indicator"] == "gdpr_80"
    assert (root / "transformed/gdpr_80/transformed_test.parquet").is_file()


def test_cluster_search_skips_infeasible_and_separates_candidate_artifacts(
    small_search_config,
):
    stages.generate_dataset(small_search_config)
    stages.split_dataset(small_search_config)
    stages.fit_preprocessor(small_search_config)
    result = stages.train_model(small_search_config)
    comparison = pd.read_parquet(
        small_search_config.artifact_dir / result.artifacts["comparison"]
    ).set_index("n_clusters")

    assert comparison.loc[2, "status"] == "success"
    assert comparison.loc[3, "status"] == "success"
    assert comparison.loc[20, "status"] == "skipped_infeasible"
    assert "20 * 10 exceeds n_train=150" in comparison.loc[20, "failure_reason"]
    assert comparison.loc[2, "constraint_violations"] == 0
    assert {
        "silhouette_score",
        "davies_bouldin_index",
        "calinski_harabasz_index",
        "minimum_cluster_size",
        "maximum_cluster_size",
        "constraint_violations",
        "training_time_seconds",
        "status",
        "failure_reason",
    }.issubset(comparison.columns)
    assert comparison.loc[2, "silhouette_status"] in {
        "available",
        "undefined: the shared silhouette sample must contain 2..n-1 clusters",
    }

    root = small_search_config.artifact_dir
    train_ids = pd.read_parquet(root / "data/train_features.parquet")[
        "id_row"
    ].tolist()
    for candidate in (2, 3):
        candidate_dir = root / "models/constrained_kmeans/k_{}".format(candidate)
        assert (candidate_dir / "model.joblib").is_file()
        assert (candidate_dir / "train_cluster_assignments.parquet").is_file()
        assert pd.read_parquet(
            candidate_dir / "train_cluster_assignments.parquet"
        )["id_row"].tolist() == train_ids
    skipped_dir = root / "models/constrained_kmeans/k_20"
    assert not (skipped_dir / "model.joblib").exists()


def test_cluster_search_builds_fitted_membership_prototypes_and_selects_without_retraining(
    small_search_config_path, small_search_config
):
    stages.generate_dataset(small_search_config)
    stages.split_dataset(small_search_config)
    stages.fit_preprocessor(small_search_config)
    trained = stages.train_model(small_search_config)
    prototypes_result = stages.build_prototypes(small_search_config)
    root = small_search_config.artifact_dir

    for candidate in (2, 3):
        base = root / "models/constrained_kmeans/k_{}".format(candidate)
        fitted = storage.read_object(
            small_search_config,
            "models/constrained_kmeans/k_{}/model.joblib".format(candidate),
        )
        assignments = pd.read_parquet(base / "train_cluster_assignments.parquet")
        prototypes = pd.read_parquet(base / "cluster_prototypes.parquet")
        assert assignments["cluster"].tolist() == fitted.labels_.tolist()
        assert prototypes["n_clients"].min() >= 10

    model_2_path = root / "models/constrained_kmeans/k_2/model.joblib"
    model_2_timestamp = model_2_path.stat().st_mtime_ns
    model_3_path = root / "models/constrained_kmeans/k_3/model.joblib"
    model_3_timestamp = model_3_path.stat().st_mtime_ns
    settings = yaml.safe_load(small_search_config_path.read_text(encoding="utf-8"))
    original_model_path = small_search_config.model_config_path

    selected_model = copy.deepcopy(small_search_config.model_config)
    selected_model["cluster_search"]["selected_n_clusters"] = 2
    selected_model_path = small_search_config.source_path.parent / "models" / "search_select_2.yml"
    selected_model_path.write_text(yaml.safe_dump(selected_model), encoding="utf-8")
    settings["model"]["config_path"] = "models/search_select_2.yml"
    selected_config_path = small_search_config.source_path.parent / "search_select_2.yml"
    selected_config_path.write_text(yaml.safe_dump(settings), encoding="utf-8")
    from cluster_representation import load_config

    selected_config = load_config(selected_config_path)
    selected_training = stages.train_model(selected_config)
    assert selected_training.artifacts["comparison"] == trained.artifacts["comparison"]
    assert model_2_path.stat().st_mtime_ns == model_2_timestamp
    assert model_3_path.stat().st_mtime_ns == model_3_timestamp
    stages.build_prototypes(selected_config)
    selected_transform = stages.transform_test(selected_config)
    assert selected_transform.artifacts["test_assignments"] == (
        "predictions/constrained_kmeans/k_2/test_cluster_assignments.parquet"
    )
    transformed_test = pd.read_parquet(
        root / selected_transform.artifacts["transformed"]
    )
    original_test = pd.read_parquet(root / selected_transform.artifacts["original"])
    selected_prototypes = pd.read_parquet(
        root / "models/constrained_kmeans/k_2/cluster_prototypes.parquet"
    ).set_index("cluster")
    replaced = transformed_test["gdpr"].eq(1)
    for feature in ("ft_1", "ft_2", "ft_3"):
        expected = transformed_test.loc[replaced, "cluster"].map(
            selected_prototypes[feature]
        )
        np.testing.assert_allclose(
            transformed_test.loc[replaced, feature].to_numpy(dtype=float),
            expected.to_numpy(dtype=float),
        )
    np.testing.assert_allclose(
        transformed_test.loc[
            ~replaced, ["ft_1", "ft_2", "ft_3"]
        ].to_numpy(dtype=float),
        original_test.loc[
            ~replaced, ["ft_1", "ft_2", "ft_3"]
        ].to_numpy(dtype=float),
    )

    extended_model = copy.deepcopy(small_search_config.model_config)
    extended_model["cluster_search"]["candidates"] = [2, 3, 4, 20]
    extended_model["cluster_search"]["selected_n_clusters"] = 2
    extended_model_path = small_search_config.source_path.parent / "models" / "search_extended.yml"
    extended_model_path.write_text(yaml.safe_dump(extended_model), encoding="utf-8")
    settings["model"]["config_path"] = "models/search_extended.yml"
    extended_config_path = small_search_config.source_path.parent / "search_extended.yml"
    extended_config_path.write_text(yaml.safe_dump(settings), encoding="utf-8")

    extended_config = load_config(extended_config_path)
    extended_training = stages.train_model(extended_config)
    assert model_2_path.stat().st_mtime_ns == model_2_timestamp
    assert (root / "models/constrained_kmeans/k_4/model.joblib").is_file()
    assert extended_training.artifacts["comparison"] != trained.artifacts["comparison"]

    stages.build_prototypes(extended_config)
    transformed = stages.transform_test(extended_config)
    assert transformed.artifacts["test_assignments"] == (
        "predictions/constrained_kmeans/k_2/test_cluster_assignments.parquet"
    )
    assert model_2_path.stat().st_mtime_ns == model_2_timestamp
    assert (
        root / "models/constrained_kmeans/k_2/cluster_prototypes.parquet"
    ).is_file()
    evaluation_result = stages.evaluate(extended_config)
    assert evaluation_result.artifacts["metrics"] == "metrics/constrained_kmeans/k_2/results.json"
    assert original_model_path.name == "search.yml"
    assert prototypes_result.summary["selected_n_clusters"] == 3


def test_cluster_search_rejects_infeasible_selected_candidate_for_downstream(
    small_search_config,
):
    stages.generate_dataset(small_search_config)
    stages.split_dataset(small_search_config)
    stages.fit_preprocessor(small_search_config)
    stages.train_model(small_search_config)
    settings = yaml.safe_load(small_search_config.source_path.read_text(encoding="utf-8"))
    model = copy.deepcopy(small_search_config.model_config)
    model["cluster_search"]["selected_n_clusters"] = 20
    model_path = small_search_config.source_path.parent / "models" / "search_select_infeasible.yml"
    model_path.write_text(yaml.safe_dump(model), encoding="utf-8")
    settings["model"]["config_path"] = "models/search_select_infeasible.yml"
    config_path = small_search_config.source_path.parent / "select_infeasible.yml"
    config_path.write_text(yaml.safe_dump(settings), encoding="utf-8")
    from cluster_representation import load_config

    config = load_config(config_path)
    with pytest.raises(ValueError, match="Selected cluster count 20 was not successfully trained"):
        stages.transform_test(config)


def test_cluster_search_records_model_fit_failures(small_search_config, monkeypatch):
    stages.generate_dataset(small_search_config)
    stages.split_dataset(small_search_config)
    stages.fit_preprocessor(small_search_config)
    original_create_model = stages.create_model

    class FailingCandidate:
        def fit(self, X):
            raise RuntimeError(
                "deliberate candidate failure for input shape {}".format(X.shape)
            )

    def create_model_with_failure(model_config):
        if model_config["parameters"]["n_clusters"] == 3:
            return FailingCandidate()
        return original_create_model(model_config)

    monkeypatch.setattr(stages, "create_model", create_model_with_failure)
    result = stages.train_model(small_search_config)
    comparison = pd.read_parquet(
        small_search_config.artifact_dir / result.artifacts["comparison"]
    ).set_index("n_clusters")
    assert comparison.loc[3, "status"] == "failed"
    assert "deliberate candidate failure" in comparison.loc[3, "failure_reason"]
    assert (small_search_config.artifact_dir / "models/constrained_kmeans/k_2/model.joblib").is_file()
    assert not (
        small_search_config.artifact_dir / "models/constrained_kmeans/k_3/model.joblib"
    ).exists()


def test_individual_stage_requires_prerequisite(small_config):
    with pytest.raises(FileNotFoundError, match="generate_dataset"):
        stages.split_dataset(small_config)


def test_artifact_fingerprint_mismatch_is_rejected(small_config):
    stage_fingerprint = storage.stage_fingerprint(small_config, "generate_dataset")
    stages.generate_dataset(small_config)
    with pytest.raises(ValueError, match="incompatible"):
        storage.require_stage(
            small_config,
            "generate_dataset",
            "different-fingerprint",
            ["data/features.parquet"],
        )
    assert stage_fingerprint


def test_notebooks_are_valid_and_thin():
    notebook_paths = sorted(__import__("pathlib").Path("notebooks").glob("*.ipynb"))
    assert len(notebook_paths) == 8
    for path in notebook_paths:
        notebook = nbformat.read(path, as_version=4)
        code = "\n".join(
            cell.source for cell in notebook.cells if cell.cell_type == "code"
        )
        assert "sys.path" not in code
        assert "read_parquet" not in code
        if path.name not in {
            "04_future_coclustering.ipynb",
            "08_constrained_coclustering.ipynb",
        }:
            assert "stages." in code


def test_config_rejects_stage_dependency_omissions(small_config_path):
    source = yaml.safe_load(small_config_path.read_text(encoding="utf-8"))
    source["pipeline"]["steps"] = ["train_model"]
    small_config_path.write_text(yaml.safe_dump(source), encoding="utf-8")
    with pytest.raises(ValueError, match="prerequisite"):
        load_config(small_config_path)


def test_metrics_handle_undefined_clustering_and_no_replacements(small_config):
    frame = pd.DataFrame(
        {"id_row": [1, 2], "ft_1": [0.0, 1.0], "ft_2": [0.0, 1.0], "ft_3": [0, 1]}
    )
    assignments = pd.DataFrame({"id_row": [1, 2], "cluster": [1, 1]})
    transformed = frame.copy()
    transformed.insert(1, "gdpr", [0, 0])
    transformed.insert(2, "cluster", [1, 1])
    metrics = evaluate(
        [[0.0, 0.0], [1.0, 1.0]],
        [1, 1],
        frame,
        assignments,
        {"gdpr_0": transformed},
        {1: 2},
        {"gdpr_0": {"requested": 0, "overall": 0, "train": 0, "test": 0}},
        small_config,
    )
    assert metrics["clustering_metrics"]["silhouette"]["value"] is None
    assert metrics["scenarios"]["gdpr_0"]["replaced_rows_mae"] is None
