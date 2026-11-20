"""Grid selection, legacy migration, and pair artifact regressions."""

from unittest.mock import patch

import numpy as np
import pytest
import yaml

from cluster_representation import load_config, storage
from cluster_representation import pipeline
from cluster_representation.robust_support import ClusterPair, candidate_feasibility
from cluster_representation.models.registry import create_model


@pytest.mark.parametrize("mode", ["auto", "fixed", "disabled"])
def test_grid_pipeline_and_candidate_reload(small_config_path, mode):
    settings = yaml.safe_load(small_config_path.read_text())
    settings["features"] = [
        dict(name="ft_{}".format(i), distribution="normal", parameters=dict(mean=i, std=1+i/3))
        for i in range(1, 7)
    ]
    name = "robust_constrained_coclustering"
    settings["models"]["run"] = [name]
    parameters = dict(seed=12345, n_init=2, iter_max=50, max_init=4, c=100, row_size_min=5)
    search = dict(enabled=mode != "disabled", row_candidates=[2, 3], column_candidates=[1, 2, 9], selection="auto" if mode == "auto" else dict(g=3, m=2))
    model = dict(name=name, parameters=parameters, cluster_search=search)
    model_path = small_config_path.parent / "models" / (name + ".yml")
    model_path.write_text(yaml.safe_dump(model))
    small_config_path.write_text(yaml.safe_dump(settings))
    config = load_config(small_config_path)
    results = pipeline.run(config)
    selection = storage.read_json(config, storage.selection_artifact_path(config))
    candidates = selection["candidate_results"]
    assert len(candidates) == (1 if mode == "disabled" else 6)
    selected = storage.selected_candidate(config)
    assert isinstance(selected, ClusterPair)
    if mode != "auto":
        assert selected == (3, 2)
    else:
        eligible = [item for item in candidates if item["eligible"]]
        winner = min(eligible, key=lambda item: (item["selection_score"], item["requested_n_clusters"], item["requested_column_clusters"]))
        assert selected == (winner["requested_n_clusters"], winner["requested_column_clusters"])
    if mode != "disabled":
        assert len(selection["excluded_candidates"]) == 2
        assert all(item["actual_column_clusters"] is None for item in candidates if item["status"] != "success")
        assert all(item["requested_column_clusters"] == 9 for item in selection["excluded_candidates"])
        assert len({item["configuration_key"] for item in candidates}) == 1
    for candidate in candidates:
        pair = ClusterPair(candidate["requested_n_clusters"], candidate["requested_column_clusters"])
        if candidate["status"] == "success":
            path = "models/{}/k_{}/model.joblib".format(name, pair)
            fitted = storage.read_object(config, path)
            assert fitted.parameters["g"] == pair.g and fitted.parameters["m"] == pair.m
            assert fitted.metadata()["n_clusters"] == candidate["actual_clusters"]
            assert fitted.metadata()["n_column_clusters"] == candidate["actual_column_clusters"]
    evaluated = next(result for result in results if result.stage == "evaluate")
    metrics = storage.read_json(config, evaluated.artifacts["metrics"])
    assert metrics["requested_row_clusters"] == selected.g
    assert metrics["requested_column_clusters"] == selected.m
    assert len(metrics["scenarios"]) == 6
    transformed = next(result for result in results if result.stage == "transform_test")
    assert str(selected) in transformed.artifacts["test_assignments"]
    for report in ("overall_comparison", "feature_evaluation"):
        assert "g={}, m={}".format(*selected) in (config.artifact_dir / "tables" / (report+".md")).read_text()
    comparison = (config.artifact_dir / "tables/cluster_selection_comparison.md").read_text()
    assert "Requested m" in comparison and "Actual feature clusters" in comparison
    with patch("cluster_representation.models.custom.robcoclust_constrained.tbcem_constrained", side_effect=AssertionError("cached candidates must not refit")):
        pipeline.run(config)


def test_legacy_migration_preserves_settings_and_rejects_conflicts():
    model = dict(name="robust_constrained_coclustering", parameters=dict(n_row_clusters=2, n_column_clusters=2, random_state=5, max_iter=40, n_init=1), constraints=dict(min_row_cluster_size=3), trimming=dict(row_fraction=0, column_fraction=0))
    fitted = create_model(model)
    assert fitted.parameters == dict(g=2, m=2, seed=5, iter_max=40, n_init=1, row_size_min=3, trim_lvl=0)
    model["parameters"]["g"] = 3
    with pytest.raises(ValueError, match="Conflicting"):
        create_model(model)


@pytest.mark.parametrize("parameters,reason", [
    (dict(g=3, m=2, row_size_min=10), "rows"),
    (dict(g=2, m=3, col_size_min=2), "columns"),
    (dict(g=2, m=2, trim_lvl=0.99), "trimming"),
])
def test_pair_feasibility(parameters, reason):
    assert reason in candidate_feasibility(parameters, np.ones((20, 4)))


@pytest.mark.parametrize("search", [
    dict(enabled=True, candidates=[2], selected_n_clusters="auto", row_candidates=[2], column_candidates=[2], selection="auto"),
    dict(enabled=True, row_candidates=[2], column_candidates=[2], selection=dict(g=3, m=2)),
    dict(enabled=False, selection="auto"),
])
def test_invalid_grid_configuration(small_config_path, search):
    settings = yaml.safe_load(small_config_path.read_text())
    settings["models"]["run"] = ["robust_constrained_coclustering"]
    model = dict(name="robust_constrained_coclustering", parameters=dict(seed=123), cluster_search=search)
    (small_config_path.parent / "models/robust_constrained_coclustering.yml").write_text(yaml.safe_dump(model))
    small_config_path.write_text(yaml.safe_dump(settings))
    with pytest.raises(ValueError):
        load_config(small_config_path)


def test_pair_ties_use_rows_then_columns():
    from cluster_representation.evaluation import resolve_cluster_selection
    candidates = []
    for g, m in ((3, 1), (2, 2), (2, 1)):
        candidates.append(dict(n_clusters=g, requested_column_clusters=m, actual_clusters=g,
            actual_column_clusters=m, status="success", minimum_cluster_size=5,
            model_parameters={"g": g, "m": m, "seed": 1},
            silhouette_score=0.5, silhouette_status="available",
            davies_bouldin_index=1, davies_bouldin_status="available",
            calinski_harabasz_index=10, calinski_harabasz_status="available"))
    selection = resolve_cluster_selection("robust_constrained_coclustering", candidates, "auto", {"min_cluster_size": 1})
    assert selection["selected_pair"] == {"g": 2, "m": 1}


def test_pair_fingerprints_include_nonsearch_settings(small_config_path):
    from cluster_representation.config import LoadedConfig
    settings = load_config(small_config_path)
    model = dict(name="robust_constrained_coclustering", parameters=dict(seed=1, trim_lvl=0.1))
    config = LoadedConfig(dict(settings), settings.source_path, model, settings.model_config_path)
    pair = ClusterPair(2, 2)
    first = storage.candidate_training_fingerprint(config, pair)
    assert first != storage.candidate_training_fingerprint(config, ClusterPair(2, 3))
    config.model_config["parameters"]["trim_lvl"] = 0.2
    assert first != storage.candidate_training_fingerprint(config, pair)
