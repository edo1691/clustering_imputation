"""Regression checks for the algorithm-preserving pipeline adapter."""

from unittest.mock import patch

import numpy as np
import pytest
import yaml

from cluster_representation import load_config
from cluster_representation import pipeline, storage
from cluster_representation.models.robust_coclustering import RobustCoclustering
from cluster_representation.models.robcoclust_constrained import tbcem_constrained


def test_adapter_matches_direct_call_and_reload(tmp_path):
    values = np.random.default_rng(25).normal(size=(90, 6))
    parameters = dict(g=2, m=2, seed=12, n_init=2, iter_max=40, c=100)
    expected = tbcem_constrained(values, **parameters)
    model = RobustCoclustering(parameters).fit(values)
    for key in ("z", "w", "mu", "sigma", "mask", "pi", "rho"):
        np.testing.assert_array_equal(expected[key], model.result_[key])
    np.testing.assert_array_equal(model.labels_, expected["z"].argmax(axis=1) + 1)
    assert expected["score"] == model.result_["score"]
    assert len(model.result_["evolution"]) == len(expected["evolution"])
    predictions = model.predict(values[:8])
    model.save(tmp_path / "model.joblib")
    loaded = RobustCoclustering.load(tmp_path / "model.joblib")
    with patch("cluster_representation.models.robust_coclustering.algorithm.tbcem_constrained", side_effect=AssertionError("prediction must not refit")):
        np.testing.assert_array_equal(predictions, loaded.predict(values[:8]))
    with pytest.raises(ValueError, match="features"):
        loaded.predict(values[:8, :2])


def test_adapter_rejects_soft_memberships():
    result = {"z": np.full((8, 2), 0.5), "w": np.ones((3, 1))}
    with patch("cluster_representation.models.robust_coclustering.algorithm.tbcem_constrained", return_value=result):
        with pytest.raises(ValueError, match="hard one-hot"):
            RobustCoclustering(dict(g=2, m=1)).fit(np.ones((8, 3)))


@pytest.mark.parametrize("mode", ["auto", "fixed_selection", "fixed_fit"])
def test_robust_pipeline_preserves_alignment_scenarios_and_outputs(small_config_path, mode):
    settings = yaml.safe_load(small_config_path.read_text())
    settings["features"] = [
        {"name": "ft_{}".format(i), "distribution": "normal", "parameters": {"mean": i, "std": 1 + i / 3}}
        for i in range(1, 7)
    ]
    settings["models"]["run"] = ["robust_coclustering"]
    settings["output"]["run_name"] = mode
    parameters = dict(m=2, seed=12345, row_size_min=5, n_init=2, iter_max=50, max_init=4, c=100)
    model_config = {"name": "robust_coclustering", "parameters": parameters}
    if mode == "fixed_fit":
        parameters["g"] = 2
    else:
        model_config["cluster_search"] = {"enabled": True, "candidates": [2, 3, 100], "selected_n_clusters": "auto" if mode == "auto" else 2}
    (small_config_path.parent / "models/robust_coclustering.yml").write_text(yaml.safe_dump(model_config))
    small_config_path.write_text(yaml.safe_dump(settings))
    config = load_config(small_config_path)
    results = pipeline.run(config)
    evaluated = results[-1]
    metrics = storage.read_json(config, evaluated.artifacts["metrics"])
    assert metrics["constraint_applicability"] == "applicable"
    assert metrics["constraint_violations"] == []
    assert set(metrics["scenarios"]) == {"gdpr_0", "gdpr_20", "gdpr_40", "gdpr_60", "gdpr_80", "gdpr_100"}
    assert metrics["model_metadata"]["n_column_clusters"] == 2
    assert metrics["model_metadata"]["column_constraint_violations"] == []
    assert list(metrics["training_feature_scales"]["feature_stds"]) == ["ft_{}".format(i) for i in range(1, 7)]
    selected = storage.selected_candidate(config)
    if mode == "fixed_fit":
        base = "models"
    else:
        selection = storage.read_json(config, storage.selection_artifact_path(config))
        assert selected in (2, 3)
        assert selection["excluded_candidates"][0]["requested_n_clusters"] == 100
        keys = [candidate["configuration_key"] for candidate in selection["candidate_results"][:2]]
        assert keys[0] == keys[1], "g must vary within a common ranking group"
        base = "models/robust_coclustering/k_{}".format(selected)
    model = storage.read_object(config, base + "/model.joblib")
    columns = storage.read_frame(config, base + "/column_labels.parquet")
    assert columns["feature"].tolist() == ["ft_{}".format(i) for i in range(1, 7)]
    np.testing.assert_array_equal(columns["column_cluster"], model.column_labels_)
    transformed = next(result for result in results if result.stage == "transform_test")
    assignments = storage.read_frame(config, transformed.artifacts["test_assignments"])
    test = storage.read_frame(config, "data/test_features.parquet")
    assert assignments["id_row"].tolist() == test["id_row"].tolist()
    preprocessor = storage.read_object(config, "preprocessing/preprocessor.joblib")
    np.testing.assert_array_equal(assignments["cluster"], model.predict(preprocessor.transform(test[columns["feature"].tolist()])))
    for scenario in metrics["scenarios"]:
        output = storage.read_frame(config, transformed.artifacts["transformed_" + scenario])
        assert output["id_row"].tolist() == test["id_row"].tolist()
        retained = output["gdpr"].eq(0)
        np.testing.assert_allclose(output.loc[retained, columns["feature"].tolist()], test.loc[retained, columns["feature"].tolist()])
    for report in ("cluster_selection_comparison", "overall_comparison", "feature_evaluation"):
        assert "robust_coclustering" in (config.artifact_dir / "tables" / (report + ".md")).read_text()
    overall = (config.artifact_dir / "tables/overall_comparison.md").read_text()
    assert "N MAE" in overall and "| MAE |" not in overall


@pytest.mark.parametrize("parameter,value", [("g", None), ("m", None), ("n_init", None), ("trim_lvl", 1), ("c", 0.5)])
def test_invalid_algorithm_parameters_fail_during_configuration(small_config_path, parameter, value):
    settings = yaml.safe_load(small_config_path.read_text())
    settings["models"]["run"] = ["robust_coclustering"]
    parameters = dict(g=2, m=2)
    parameters[parameter] = value
    model_path = small_config_path.parent / "models/robust_coclustering.yml"
    model_path.write_text(yaml.safe_dump({"name": "robust_coclustering", "parameters": parameters}))
    small_config_path.write_text(yaml.safe_dump(settings))
    with pytest.raises(ValueError):
        load_config(small_config_path)
