import copy
import json

import nbformat
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
