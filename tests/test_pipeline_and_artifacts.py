import copy
import json

import nbformat
import numpy as np
import pandas as pd
import pytest
import yaml

from cluster_representation import load_config
from cluster_representation import pipeline, reporting, stages, storage, transformation
from cluster_representation.evaluation import evaluate, resolve_cluster_selection
from cluster_representation.models.constrained_kmeans import ConstrainedKMeans
from cluster_representation.nonconsent import scenario_percentages


def test_end_to_end_pipeline_writes_reusable_artifacts(small_config):
    results = pipeline.run(small_config)
    assert small_config.models_run == ["constrained_kmeans"]
    assert [result.stage for result in results] == small_config["pipeline"]["steps"]
    root = small_config.artifact_dir
    assert (root / "manifest.json").is_file()
    assert (root / "resolved_config.yml").is_file()
    assert (root / "environment.json").is_file()
    assert (root / "prototypes/train_cluster_assignments.parquet").is_file()
    assert (root / "prototypes/cluster_prototypes.parquet").is_file()
    assert (root / "predictions/test_cluster_assignments.parquet").is_file()
    assert (root / "metrics/results.json").is_file()
    comparison_path = root / "tables/overall_comparison.md"
    assert comparison_path.is_file()
    transformed = pd.read_parquet(
        root
        / "models/constrained_kmeans/transformed/gdpr_20/transformed_test.parquet"
    )
    originals = pd.read_parquet(
        root
        / "models/constrained_kmeans/transformed/gdpr_20/original_test.parquet"
    )
    assert transformed["id_row"].tolist() == originals["id_row"].tolist()
    assert len(transformed) == 50

    metrics = json.loads((root / "metrics/results.json").read_text(encoding="utf-8"))
    assert len(metrics["scenarios"]) == 6
    assert metrics["scenarios"]["gdpr_0"]["replaced_rows_mae"] is None
    assert metrics["scenarios"]["gdpr_0"]["replaced_rows_mae_status"] == "unavailable_no_replaced_rows"
    assert metrics["scenarios"]["gdpr_0"]["replaced_rows_mape"] is None
    assert metrics["scenarios"]["gdpr_0"]["replaced_rows_mape_status"] == (
        "unavailable_no_replaced_rows"
    )
    assert "overall_test_mae" in metrics["scenarios"]["gdpr_20"]
    assert "per_feature_mae" in metrics["scenarios"]["gdpr_20"]
    assert "Replaced-row MAPE excludes cells with zero original values" in metrics[
        "notes"
    ][1]
    markdown = comparison_path.read_text(encoding="utf-8")
    assert markdown.startswith("### Overall comparison of evaluated methods\n\n")
    assert (
        "| Method | NC (%) | AP | F1 | MAE | MAPE (%) | \\|G\\| | k̄ | Time (s) |"
        in markdown
    )
    table_rows = [
        line
        for line in markdown.splitlines()
        if line.startswith("|")
        and not line.startswith("| Method")
        and not line.startswith("|--------")
    ]
    assert len(table_rows) == 6
    for line, percent in zip(table_rows, (0, 20, 40, 60, 80, 100)):
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        scenario = metrics["scenarios"]["gdpr_{}".format(percent)]
        assert cells[:4] == [
            "constrained_kmeans (k=3)",
            str(percent),
            "—",
            "—",
        ]
        if scenario["replaced_rows_mae"] is None:
            assert cells[4] == "—"
        else:
            assert float(cells[4]) == pytest.approx(
                scenario["replaced_rows_mae"]
            )
        if scenario["replaced_rows_mape"] is None:
            assert cells[5] == "—"
        else:
            assert float(cells[5]) == pytest.approx(
                scenario["replaced_rows_mape"]
            )
        assert cells[6:] == [
            "3",
            str(int(metrics["average_training_cluster_size"])),
            str(metrics["timings_seconds"]["fitting"]),
        ]
    assert "MAE and MAPE are calculated only for replaced rows." in markdown
    assert "MAPE excludes cells with zero original values." in markdown
    assert "MAE averages features on their existing scales." in markdown
    assert "Baseline" not in markdown
    assert "CC |" not in markdown
    assert "RCCC" not in markdown
    combined = reporting.combined_evaluation_report(small_config)
    assert len(combined) == 6
    assert combined["Clusters"].eq(3).all()
    assert combined.loc[
        combined["Non-consent (%)"].eq(0), "MAE (replaced rows) ↓"
    ].iloc[0] == "N/A"
    assert (root / "metrics/combined_evaluation.csv").is_file()
    assert (root / "metrics/combined_evaluation.parquet").is_file()
    assert not (root / "models/spectral_coclustering").exists()


def test_overall_comparison_uses_configured_range(small_config_path):
    default_config = load_config("config/default.yml")
    assert scenario_percentages(default_config) == list(range(10, 101, 10))

    settings = yaml.safe_load(small_config_path.read_text(encoding="utf-8"))
    settings["nonconsent"].update(
        {"start_percent": 10, "end_percent": 30, "step_percent": 10}
    )
    settings["output"]["run_name"] = "changed_scenario_range"
    small_config_path.write_text(yaml.safe_dump(settings), encoding="utf-8")
    config = load_config(small_config_path)

    pipeline.run(config)

    root = config.artifact_dir
    metrics = json.loads((root / "metrics/results.json").read_text(encoding="utf-8"))
    markdown = (root / "tables/overall_comparison.md").read_text(encoding="utf-8")
    rows = [
        [cell.strip() for cell in line.strip("|").split("|")]
        for line in markdown.splitlines()
        if line.startswith("|")
        and not line.startswith("| Method")
        and not line.startswith("|--------")
    ]
    assert [int(row[1]) for row in rows] == [10, 20, 30]
    for row in rows:
        scenario = metrics["scenarios"]["gdpr_{}".format(row[1])]
        assert row[0] == "constrained_kmeans (k=3)"
        assert float(row[4]) == pytest.approx(scenario["replaced_rows_mae"])
        if scenario["replaced_rows_mape"] is None:
            assert row[5] == "—"
        else:
            assert float(row[5]) == pytest.approx(scenario["replaced_rows_mape"])

    indicators = storage.read_frame(config, "data/nonconsent.parquet")
    assert indicators[["gdpr_10", "gdpr_20", "gdpr_30"]].sum().tolist() == [
        20,
        40,
        60,
    ]
    assert (indicators["gdpr_10"] <= indicators["gdpr_20"]).all()
    assert (indicators["gdpr_20"] <= indicators["gdpr_30"]).all()


def test_combined_report_aligns_scenarios_and_reuses_fitted_candidates(
    small_search_config, monkeypatch
):
    results = pipeline.run(small_search_config)
    train_result = next(result for result in results if result.stage == "train_model")
    evaluation_result = next(result for result in results if result.stage == "evaluate")
    root = small_search_config.artifact_dir
    markdown = (root / "tables/overall_comparison.md").read_text(encoding="utf-8")
    assert "| constrained_kmeans (k=3) | 20 |" in markdown
    assert "Baseline" not in markdown
    assert "CC |" not in markdown
    assert "RCCC" not in markdown
    assert evaluation_result.artifacts["overall_comparison"] == (
        "tables/overall_comparison.md"
    )
    comparison = pd.read_parquet(root / train_result.artifacts["comparison"])
    successful = comparison.loc[comparison["status"].eq("success")]

    def fail_if_fit(*args, **kwargs):
        raise AssertionError("Reporting must not fit or retrain models")

    monkeypatch.setattr(ConstrainedKMeans, "fit", fail_if_fit)
    display = reporting.combined_evaluation_report(small_search_config)

    exported = pd.read_parquet(root / "metrics/combined_evaluation.parquet")
    exported_csv = pd.read_csv(root / "metrics/combined_evaluation.csv")
    assert len(exported) == len(successful) * 6
    assert not exported.duplicated(
        ["model", "n_clusters", "nonconsent_percent"]
    ).any()
    pd.testing.assert_frame_equal(
        exported,
        exported_csv,
        check_dtype=False,
        check_exact=False,
    )
    assert display.columns.tolist() == [
        "Model",
        "Clusters",
        "Constraint applicability",
        "Non-consent (%)",
        "Smallest cluster",
        "Silhouette ↑",
        "Davies–Bouldin ↓",
        "Calinski–Harabasz ↑",
        "MAE ↓",
        "MAE (replaced rows) ↓",
        "Realized test non-consent (%)",
    ]

    candidate = 2
    row = exported.loc[
        exported["n_clusters"].eq(candidate)
        & exported["nonconsent_percent"].eq(20)
    ].iloc[0]
    comparison_row = comparison.loc[comparison["n_clusters"].eq(candidate)].iloc[0]
    assert row["silhouette_score"] == pytest.approx(
        comparison_row["silhouette_score"]
    )
    assert row["constraint_applicability"] == "applicable"
    assert row["davies_bouldin_index"] == pytest.approx(
        comparison_row["davies_bouldin_index"]
    )
    assert row["calinski_harabasz_index"] == pytest.approx(
        comparison_row["calinski_harabasz_index"]
    )
    assert row["smallest_cluster_size"] == comparison_row["minimum_cluster_size"]

    test = storage.read_frame(small_search_config, "data/test_features.parquet")
    indicators = storage.read_frame(
        small_search_config, "data/test_indicators.parquet"
    )
    preprocessor = storage.read_object(
        small_search_config, "preprocessing/preprocessor.joblib"
    )
    model = storage.read_object(
        small_search_config,
        "models/constrained_kmeans/k_2/model.joblib",
    )
    prototypes = storage.read_frame(
        small_search_config,
        "models/constrained_kmeans/k_2/cluster_prototypes.parquet",
    )
    assignments = transformation.assign_test_clusters(
        test, small_search_config, preprocessor, model
    )["assignments"]
    transformed = transformation.transform_test(
        test,
        indicators[["id_row", "gdpr_20"]],
        prototypes,
        assignments,
        small_search_config,
        "gdpr_20",
    )["transformed"]
    feature_names = [feature["name"] for feature in small_search_config["features"]]
    errors = np.abs(
        transformed[feature_names].to_numpy(dtype=float)
        - test[feature_names].to_numpy(dtype=float)
    )
    replaced = transformed["gdpr"].eq(1).to_numpy()
    assert row["mae"] == pytest.approx(float(errors.mean()))
    assert row["mae_replaced_rows"] == pytest.approx(
        float(errors[replaced].mean())
    )
    assert row["realized_test_nonconsent_percent"] == pytest.approx(
        float(indicators["gdpr_20"].mean() * 100.0)
    )
    zero_scenario = exported.loc[
        exported["n_clusters"].eq(candidate)
        & exported["nonconsent_percent"].eq(0)
    ].iloc[0]
    assert zero_scenario["mae_replaced_rows"] != zero_scenario[
        "mae_replaced_rows"
    ]
    assert display.loc[
        display["Clusters"].eq(candidate)
        & display["Non-consent (%)"].eq(0),
        "MAE (replaced rows) ↓",
    ].iloc[0] == "N/A"


def test_cluster_selection_report_uses_saved_candidate_metadata_and_ranks_metrics(
    small_search_config, monkeypatch
):
    results = pipeline.run(small_search_config)
    train_result = next(result for result in results if result.stage == "train_model")
    evaluation_result = next(result for result in results if result.stage == "evaluate")
    root = small_search_config.artifact_dir
    comparison = pd.read_parquet(root / train_result.artifacts["comparison"])
    successful = comparison.loc[comparison["status"].eq("success")]

    report_path = root / "tables/cluster_selection_comparison.md"
    assert report_path.is_file()
    assert evaluation_result.artifacts["cluster_selection_comparison"] == (
        "tables/cluster_selection_comparison.md"
    )
    markdown = report_path.read_text(encoding="utf-8")
    rows = [
        [cell.strip() for cell in line.strip("|").split("|")]
        for line in markdown.splitlines()
        if line.startswith("| constrained_kmeans")
    ]
    assert len(rows) == len(comparison)
    assert [int(row[1]) for row in rows] == comparison["n_clusters"].tolist()
    assert len({(row[0], row[1]) for row in rows}) == len(rows)

    by_k = {int(row[1]): row for row in rows}
    for candidate in successful.to_dict(orient="records"):
        requested_k = int(candidate["n_clusters"])
        metadata = storage.read_json(
            small_search_config,
            "models/constrained_kmeans/k_{}/metadata.json".format(requested_k),
        )
        row = by_k[requested_k]
        assert row[2] == str(metadata["n_clusters"])
        assert row[8] == str(metadata["training_time_seconds"])
        assert row[7] == (
            "Yes"
            if requested_k
            == small_search_config.model_config["cluster_search"]["selected_n_clusters"]
            else "No"
        )
        for index, value_key, status_key, maximize in (
            (3, "silhouette_score", "silhouette_status", True),
            (4, "davies_bouldin_index", "davies_bouldin_status", False),
            (
                5,
                "calinski_harabasz_index",
                "calinski_harabasz_status",
                True,
            ),
        ):
            value = (
                metadata[value_key]
                if metadata[status_key] == "available"
                else None
            )
            text = row[index]
            assert (text == "—") == (value is None)
            if value is not None:
                rendered = text.removeprefix("**").removesuffix("**")
                assert float(rendered) == value
                assert text.startswith("**") == (
                    value
                    == (
                        successful[value_key].max()
                        if maximize
                        else successful[value_key].min()
                    )
                )

    assert "Bold values identify the best result" in markdown
    assert "Higher silhouette and Calinski–Harabasz are better." in markdown
    assert "Lower Davies–Bouldin is better." in markdown

    def fail_if_fit(*args, **kwargs):
        raise AssertionError("Reporting must not retrain candidate models")

    monkeypatch.setattr(ConstrainedKMeans, "fit", fail_if_fit)
    evaluation_metrics = storage.read_json(
        small_search_config, evaluation_result.artifacts["metrics"]
    )
    reporting.export_cluster_selection_comparison(
        small_search_config, evaluation_metrics
    )


def test_cluster_selection_report_marks_all_metric_ties():
    rows = [
        {
            "method": "test_model",
            "configuration_key": "{}",
            "silhouette": 0.4,
            "davies_bouldin": 1.0,
            "calinski_harabasz": 12.0,
        },
        {
            "method": "test_model",
            "configuration_key": "{}",
            "silhouette": 0.4,
            "davies_bouldin": 1.0,
            "calinski_harabasz": 12.0,
        },
    ]
    reporting._mark_metric_winners(rows, "silhouette", maximize=True)
    reporting._mark_metric_winners(rows, "davies_bouldin", maximize=False)
    reporting._mark_metric_winners(rows, "calinski_harabasz", maximize=True)
    assert all(row["silhouette_winner"] for row in rows)
    assert all(row["davies_bouldin_winner"] for row in rows)
    assert all(row["calinski_harabasz_winner"] for row in rows)

    separate_configuration = [
        {
            "method": "test_model",
            "configuration_key": "first",
            "silhouette": 0.1,
            "davies_bouldin": 2.0,
            "calinski_harabasz": 4.0,
        },
        {
            "method": "test_model",
            "configuration_key": "first",
            "silhouette": 0.2,
            "davies_bouldin": 1.0,
            "calinski_harabasz": 5.0,
        },
        {
            "method": "test_model",
            "configuration_key": "second",
            "silhouette": 0.9,
            "davies_bouldin": 3.0,
            "calinski_harabasz": 2.0,
        },
        {
            "method": "test_model",
            "configuration_key": "second",
            "silhouette": 0.5,
            "davies_bouldin": 0.5,
            "calinski_harabasz": 8.0,
        },
    ]
    reporting._mark_metric_winners(
        separate_configuration, "silhouette", maximize=True
    )
    reporting._mark_metric_winners(
        separate_configuration, "davies_bouldin", maximize=False
    )
    reporting._mark_metric_winners(
        separate_configuration, "calinski_harabasz", maximize=True
    )
    assert [
        row["silhouette_winner"] for row in separate_configuration
    ] == [False, True, True, False]
    assert [
        row["davies_bouldin_winner"] for row in separate_configuration
    ] == [False, True, False, True]
    assert [
        row["calinski_harabasz_winner"] for row in separate_configuration
    ] == [False, True, False, True]


def test_auto_cluster_selection_uses_average_ranks_and_fixed_selection_is_preserved():
    candidates = []
    for k, silhouette, davies_bouldin, calinski_harabasz in (
        (5, 0.9, 0.9, 10.0),
        (6, 0.8, 1.0, 30.0),
        (7, 0.7, 0.8, 20.0),
        (8, 0.95, 0.5, 40.0),
    ):
        candidates.append(
            {
                "n_clusters": k,
                "status": "success",
                "model_parameters": json.dumps(
                    {"n_clusters": k, "random_state": 123}
                ),
                "constraints": json.dumps({"min_cluster_size": 10}),
                "minimum_cluster_size": 10 if k != 8 else 2,
                "constraint_violations": 0 if k != 8 else 1,
                "constraint_violation_clusters": [] if k != 8 else [1],
                "silhouette_score": silhouette,
                "silhouette_status": "available",
                "davies_bouldin_index": davies_bouldin,
                "davies_bouldin_status": "available",
                "calinski_harabasz_index": calinski_harabasz,
                "calinski_harabasz_status": "available",
            }
        )
    constraints = {"min_cluster_size": 10}

    automatic = resolve_cluster_selection("test_method", candidates, "auto", constraints)

    eligible = automatic["candidate_results"][:3]
    assert [candidate["selection_score"] for candidate in eligible] == [
        2.0,
        2.0,
        2.0,
    ]
    assert automatic["selected_n_clusters"] == 5
    assert automatic["candidate_results"][3]["eligible"] is False
    assert "hard constraint violations" in automatic["candidate_results"][3][
        "exclusion_reasons"
    ][0]
    assert automatic["excluded_candidates"][0]["requested_n_clusters"] == 8

    fixed = resolve_cluster_selection("test_method", candidates, 7, constraints)
    assert fixed["selection_mode"] == "fixed"
    assert fixed["selected_n_clusters"] == 7
    assert fixed["configured_selected_n_clusters"] == 7


def test_auto_cluster_selection_reports_no_eligible_candidates():
    failed = {
        "n_clusters": 5,
        "status": "failed",
        "failure_reason": "fit failed",
        "silhouette_status": "unavailable",
        "davies_bouldin_status": "unavailable",
        "calinski_harabasz_status": "unavailable",
    }

    with pytest.raises(
        ValueError,
        match=r"no eligible candidates for constrained_kmeans.*k=5: training status is failed",
    ):
        resolve_cluster_selection(
            "constrained_kmeans", [failed], "auto", {"min_cluster_size": 10}
        )


def test_auto_selection_routes_all_downstream_artifacts_through_resolved_model(
    small_search_config_path,
):
    settings = yaml.safe_load(
        small_search_config_path.read_text(encoding="utf-8")
    )
    model = copy.deepcopy(
        yaml.safe_load(
            (small_search_config_path.parent / "models/search.yml").read_text(
                encoding="utf-8"
            )
        )
    )
    model["cluster_search"]["selected_n_clusters"] = "auto"
    auto_model_path = small_search_config_path.parent / "models/search_auto.yml"
    auto_model_path.write_text(yaml.safe_dump(model), encoding="utf-8")
    settings["models"]["configs"]["constrained_kmeans"] = "models/search_auto.yml"
    settings["output"]["run_name"] = "search_auto_run"
    auto_config_path = small_search_config_path.parent / "search_auto.yml"
    auto_config_path.write_text(yaml.safe_dump(settings), encoding="utf-8")
    from cluster_representation import load_config

    auto_config = load_config(auto_config_path)
    results = pipeline.run(auto_config)
    train_result = next(result for result in results if result.stage == "train_model")
    selection = storage.read_json(auto_config, train_result.artifacts["selection"])
    selected_k = selection["selected_n_clusters"]
    assert selection["selection_mode"] == "auto"
    assert selection["configured_selected_n_clusters"] == "auto"
    assert [item["requested_n_clusters"] for item in selection["candidate_results"]] == [
        2,
        3,
        20,
    ]
    assert selection["excluded_candidates"][0]["requested_n_clusters"] == 20
    assert "skipped_infeasible" in selection["excluded_candidates"][0]["reasons"][0]

    root = auto_config.artifact_dir
    resolved_config = yaml.safe_load(
        (root / "resolved_config.yml").read_text(encoding="utf-8")
    )
    assert (
        resolved_config["resolved_model_configs"]["constrained_kmeans"][
            "cluster_search"
        ][
            "selected_n_clusters"
        ]
        == "auto"
    )
    assert (root / "models/constrained_kmeans/k_{}/cluster_prototypes.parquet".format(
        selected_k
    )).is_file()
    assert (root / "predictions/constrained_kmeans/k_{}/test_cluster_assignments.parquet".format(
        selected_k
    )).is_file()
    assert (root / "metrics/constrained_kmeans/k_{}/results.json".format(
        selected_k
    )).is_file()
    assignments = pd.read_parquet(
        root
        / "models/constrained_kmeans/k_{}/train_cluster_assignments.parquet".format(
            selected_k
        )
    )
    fitted_model = storage.read_object(
        auto_config,
        "models/constrained_kmeans/k_{}/model.joblib".format(selected_k),
    )
    assert assignments["cluster"].tolist() == fitted_model.labels_.tolist()

    report = (root / "tables/cluster_selection_comparison.md").read_text(
        encoding="utf-8"
    )
    report_rows = [
        [cell.strip() for cell in line.strip("|").split("|")]
        for line in report.splitlines()
        if line.startswith("| constrained_kmeans")
    ]
    assert len(report_rows) == 3
    assert next(row for row in report_rows if int(row[1]) == selected_k)[7] == "Yes"
    assert next(row for row in report_rows if row[1] == "20")[6] == "—"
    assert "Automatic selection averages" in report


def test_spectral_coclustering_runs_fixed_and_auto_pipeline_without_retraining(
    small_config_path,
):
    settings = yaml.safe_load(
        small_config_path.read_text(encoding="utf-8")
    )
    model_config = {
        "name": "spectral_coclustering",
        "parameters": {"random_state": 12345, "n_init": 2},
        "cluster_search": {
            "enabled": True,
            "candidates": [2, 3],
            "selected_n_clusters": 2,
        },
    }
    model_path = small_config_path.parent / "models/spectral.yml"
    model_path.write_text(yaml.safe_dump(model_config), encoding="utf-8")
    settings["models"]["run"] = ["spectral_coclustering"]
    settings["models"]["configs"] = {"spectral_coclustering": "models/spectral.yml"}
    settings["output"]["run_name"] = "spectral_pipeline"
    config_path = small_config_path.parent / "spectral_run.yml"
    config_path.write_text(yaml.safe_dump(settings), encoding="utf-8")
    from cluster_representation import load_config

    fixed_config = load_config(config_path)
    fixed_results = pipeline.run(fixed_config)
    fixed_training = next(
        result for result in fixed_results if result.stage == "train_model"
    )
    assert fixed_training.summary["selected_n_clusters"] == 2
    root = fixed_config.artifact_dir
    model_path_k2 = root / "models/spectral_coclustering/k_2/model.joblib"
    model_timestamp = model_path_k2.stat().st_mtime_ns
    for candidate in (2, 3):
        base = root / "models/spectral_coclustering/k_{}".format(candidate)
        assert (base / "model.joblib").is_file()
        assert (base / "model_preprocessor.joblib").is_file()
        assert (base / "column_labels.parquet").is_file()
        assignments = pd.read_parquet(base / "train_cluster_assignments.parquet")
        model = storage.read_object(
            fixed_config,
            "models/spectral_coclustering/k_{}/model.joblib".format(candidate),
        )
        train = pd.read_parquet(root / "data/train_features.parquet")
        shared_preprocessor = storage.read_object(
            fixed_config, "preprocessing/preprocessor.joblib"
        )
        training_matrix = shared_preprocessor.transform(
            train[[feature["name"] for feature in fixed_config["features"]]]
        )
        np.testing.assert_allclose(
            model.feature_min_, training_matrix.min(axis=0)
        )
        assert assignments["id_row"].tolist() == train["id_row"].tolist()
        assert assignments["cluster"].tolist() == model.labels_.tolist()
        metadata = storage.read_json(
            fixed_config,
            "models/spectral_coclustering/k_{}/metadata.json".format(candidate),
        )
        assert metadata["constraint_applicability"] == "not applicable"
        assert metadata["assignment_strategy"] == model.assignment_strategy
        assert metadata["minimum_cluster_size"] < 100
        columns = pd.read_parquet(base / "column_labels.parquet")
        np.testing.assert_array_equal(columns["column_cluster"], model.column_labels_)

    model_config["cluster_search"]["selected_n_clusters"] = "auto"
    model_path.write_text(yaml.safe_dump(model_config), encoding="utf-8")
    auto_config = load_config(config_path)
    auto_results = pipeline.run(auto_config)
    auto_training = next(
        result for result in auto_results if result.stage == "train_model"
    )
    selection = storage.read_json(auto_config, auto_training.artifacts["selection"])
    eligible = [
        item for item in selection["candidate_results"] if item["eligible"]
    ]
    assert selection["selection_mode"] == "auto"
    assert selection["configured_selected_n_clusters"] == "auto"
    assert selection["selected_n_clusters"] == min(
        eligible, key=lambda item: (item["selection_score"], item["requested_n_clusters"])
    )["requested_n_clusters"]
    assert model_path_k2.stat().st_mtime_ns == model_timestamp

    selected = selection["selected_n_clusters"]
    test = pd.read_parquet(root / "data/test_features.parquet")
    predicted = pd.read_parquet(
        root
        / "predictions/spectral_coclustering/k_{}/test_cluster_assignments.parquet".format(
            selected
        )
    )
    assert predicted["id_row"].tolist() == test["id_row"].tolist()
    assert (root / "models/spectral_coclustering/k_{}/cluster_prototypes.parquet".format(
        selected
    )).is_file()
    metrics_path = "metrics/spectral_coclustering/k_{}/results.json".format(selected)
    metrics = storage.read_json(auto_config, metrics_path)
    assert metrics["constraint_applicability"] == "not applicable"
    assert metrics["assignment_strategy"].startswith("nearest_training_centroid")
    assert metrics["clustering_metric_feature_space"].startswith(
        "shared fitted preprocessing output"
    )
    assert metrics["classification_metrics"]["average_precision"] == {
        "value": None,
        "status": "unavailable_no_ground_truth_labels",
    }
    assert len(metrics["scenarios"]) == 6
    for scenario in metrics["scenarios"].values():
        assert "replaced_rows_mae" in scenario
        assert "replaced_rows_mape" in scenario

    report_path = root / "tables/cluster_selection_comparison.md"
    report = report_path.read_text(encoding="utf-8")
    report_rows = [
        [cell.strip() for cell in line.strip("|").split("|")]
        for line in report.splitlines()
        if line.startswith("| spectral_coclustering")
    ]
    assert len(report_rows) == 2
    assert next(row for row in report_rows if int(row[1]) == selected)[7] == "Yes"
    assert "Constraint applicability: not applicable." in report
    overall = (root / "tables/overall_comparison.md").read_text(encoding="utf-8")
    assert len(
        [
            line
            for line in overall.splitlines()
            if line.startswith("| spectral_coclustering")
        ]
    ) == 6
    assert "Constraints: not applicable" in overall
    combined = reporting.combined_evaluation_report(auto_config)
    assert (combined["Constraint applicability"] == "not applicable").all()
    assert len(combined) == len(eligible) * 6


def test_pipeline_runs_selected_models_in_order_and_combines_run_reports(
    small_search_config_path, monkeypatch
):
    settings = yaml.safe_load(
        small_search_config_path.read_text(encoding="utf-8")
    )
    spectral_config_path = small_search_config_path.parent / "models/spectral.yml"
    spectral_model = {
        "name": "spectral_coclustering",
        "parameters": {"random_state": 12345, "n_init": 2},
        "cluster_search": {
            "enabled": True,
            "candidates": [2, 3],
            "selected_n_clusters": "auto",
        },
    }
    spectral_config_path.write_text(yaml.safe_dump(spectral_model), encoding="utf-8")
    settings["models"]["run"] = ["constrained_kmeans", "spectral_coclustering"]
    settings["models"]["configs"]["spectral_coclustering"] = "models/spectral.yml"
    settings["nonconsent"].update(
        {"start_percent": 10, "end_percent": 100, "step_percent": 10}
    )
    settings["output"]["run_name"] = "multi_model_run"
    config_path = small_search_config_path.parent / "multi_model.yml"
    config_path.write_text(yaml.safe_dump(settings), encoding="utf-8")
    from cluster_representation import load_config

    config = load_config(config_path)
    original_transform = transformation.transform_test
    transformed_scenarios = []

    def track_transform(*args, **kwargs):
        transformed_scenarios.append(
            (args[4].model_config["name"], args[5])
        )
        return original_transform(*args, **kwargs)

    monkeypatch.setattr(transformation, "transform_test", track_transform)
    results = pipeline.run(config)
    model_training = [
        result
        for result in results
        if result.stage == "train_model"
    ]
    assert [result.config.model_config["name"] for result in model_training] == [
        "constrained_kmeans",
        "spectral_coclustering",
    ]
    assert sum(result.stage == "generate_dataset" for result in results) == 1
    assert sum(result.stage == "fit_preprocessor" for result in results) == 1
    assert len([result for result in results if result.stage == "evaluate"]) == 2

    root = config.artifact_dir
    test = pd.read_parquet(root / "data/test_features.parquet")
    indicators = pd.read_parquet(root / "data/test_indicators.parquet")
    scenario_names = ["gdpr_{}".format(percent) for percent in range(10, 101, 10)]
    assert len(transformed_scenarios) == 20
    for model_name in ("constrained_kmeans", "spectral_coclustering"):
        assert [
            scenario for model, scenario in transformed_scenarios if model == model_name
        ] == scenario_names
    for model_name in ("constrained_kmeans", "spectral_coclustering"):
        model_config = config.for_model(model_name)
        selected_k = storage.selected_candidate(model_config)
        assert (root / "models/{}/search_{}/comparison.parquet".format(
            model_name,
            storage.cluster_search_fingerprint(model_config)[:12],
        )).is_file()
        assert (root / "metrics/{}/k_{}".format(
            model_name,
            selected_k,
        ) / "results.json").is_file()
        for scenario_name in scenario_names:
            base = root / "models/{}/k_{}/transformed/{}".format(
                model_name, selected_k, scenario_name
            )
            transformed = pd.read_parquet(base / "transformed_test.parquet")
            original = pd.read_parquet(base / "original_test.parquet")
            assert transformed["id_row"].tolist() == test["id_row"].tolist()
            assert original["id_row"].tolist() == test["id_row"].tolist()
            assert transformed["gdpr"].tolist() == indicators[
                scenario_name
            ].tolist()
            replaced = transformed["gdpr"].eq(1).to_numpy()
            np.testing.assert_allclose(
                transformed.loc[~replaced, ["ft_1", "ft_2", "ft_3"]].to_numpy(
                    dtype=float
                ),
                original.loc[~replaced, ["ft_1", "ft_2", "ft_3"]].to_numpy(
                    dtype=float
                ),
            )
    overall = (root / "tables/overall_comparison.md").read_text(encoding="utf-8")
    overall_rows = [
        line
        for line in overall.splitlines()
        if line.startswith("| constrained_kmeans")
        or line.startswith("| spectral_coclustering")
    ]
    assert len(overall_rows) == 20
    assert sum(line.startswith("| constrained_kmeans") for line in overall_rows) == 10
    assert sum(line.startswith("| spectral_coclustering") for line in overall_rows) == 10
    for model_name in ("constrained_kmeans", "spectral_coclustering"):
        model_rows = [
            line for line in overall_rows if line.startswith("| {}".format(model_name))
        ]
        assert [
            int(line.split("|")[2].strip()) for line in model_rows
        ] == list(range(10, 101, 10))
    cluster_report = (
        root / "tables/cluster_selection_comparison.md"
    ).read_text(encoding="utf-8")
    assert any(
        line.startswith("| constrained_kmeans")
        for line in cluster_report.splitlines()
    )
    assert any(
        line.startswith("| spectral_coclustering")
        for line in cluster_report.splitlines()
    )
    assert "Selection mode: auto." in cluster_report


def test_transform_stage_processes_every_scenario_from_original_test_rows(small_config):
    pipeline.run(small_config)
    root = small_config.artifact_dir
    assignments_before = pd.read_parquet(
        root / "predictions/test_cluster_assignments.parquet"
    )
    model_path = root / "models/model.joblib"
    model_timestamp = model_path.stat().st_mtime_ns
    result = stages.transform_test(small_config)

    assignments_after = pd.read_parquet(
        root / "predictions/test_cluster_assignments.parquet"
    )
    pd.testing.assert_frame_equal(assignments_before, assignments_after)
    assert model_path.stat().st_mtime_ns == model_timestamp
    scenarios = ["gdpr_{}".format(percent) for percent in range(0, 101, 20)]
    assert result.summary["scenarios"] == scenarios
    test = pd.read_parquet(root / "data/test_features.parquet")
    indicators = pd.read_parquet(root / "data/test_indicators.parquet")
    feature_names = [feature["name"] for feature in small_config["features"]]
    for scenario in scenarios:
        transformed_path = result.artifacts["transformed_{}".format(scenario)]
        original_path = result.artifacts["original_{}".format(scenario)]
        assert (root / transformed_path).is_file()
        assert (root / original_path).is_file()
        transformed = pd.read_parquet(root / transformed_path)
        original = pd.read_parquet(root / original_path)
        pd.testing.assert_frame_equal(
            original[["id_row"] + feature_names],
            test[["id_row"] + feature_names],
        )
        assert transformed["id_row"].tolist() == test["id_row"].tolist()
        assert transformed["gdpr"].tolist() == indicators[scenario].tolist()
        replaced = transformed["gdpr"].eq(1).to_numpy()
        np.testing.assert_allclose(
            transformed.loc[~replaced, feature_names].to_numpy(dtype=float),
            original.loc[~replaced, feature_names].to_numpy(dtype=float),
        )


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
    settings["models"]["configs"]["constrained_kmeans"] = "models/search_select_2.yml"
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
        root / selected_transform.artifacts["transformed_gdpr_20"]
    )
    original_test = pd.read_parquet(
        root / selected_transform.artifacts["original_gdpr_20"]
    )
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
    settings["models"]["configs"]["constrained_kmeans"] = "models/search_extended.yml"
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
    settings["models"]["configs"]["constrained_kmeans"] = (
        "models/search_select_infeasible.yml"
    )
    config_path = small_search_config.source_path.parent / "select_infeasible.yml"
    config_path.write_text(yaml.safe_dump(settings), encoding="utf-8")
    from cluster_representation import load_config

    config = load_config(config_path)
    with pytest.raises(ValueError, match="Selected cluster count 20 was not successfully trained"):
        stages.transform_test(config)


def test_cluster_search_records_model_fit_failures(small_search_config, monkeypatch):
    model_config = copy.deepcopy(small_search_config.model_config)
    model_config["cluster_search"]["selected_n_clusters"] = 2
    small_search_config.model_config_path.write_text(
        yaml.safe_dump(model_config), encoding="utf-8"
    )
    from cluster_representation import load_config

    small_search_config = load_config(small_search_config.source_path)
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
            "07_evaluate.ipynb",
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
    assert metrics["scenarios"]["gdpr_0"]["replaced_rows_mae_status"] == (
        "unavailable_no_replaced_rows"
    )
    assert metrics["scenarios"]["gdpr_0"]["replaced_rows_mape"] is None
    assert metrics["scenarios"]["gdpr_0"]["replaced_rows_mape_status"] == (
        "unavailable_no_replaced_rows"
    )
    assert metrics["scenarios"]["gdpr_0"]["replaced_rows_mape_valid_cells"] == 0
    assert metrics["scenarios"]["gdpr_0"][
        "replaced_rows_mape_excluded_zero_cells"
    ] == 0


def test_replaced_row_mae_and_mape_exclude_nonreplaced_rows(small_config):
    original = pd.DataFrame(
        {
            "id_row": [1, 2],
            "ft_1": [10.0, 1.0],
            "ft_2": [20.0, 1.0],
            "ft_3": [0.0, 1.0],
        }
    )
    assignments = pd.DataFrame({"id_row": [1, 2], "cluster": [1, 1]})
    transformed = original.copy()
    transformed.loc[0, ["ft_1", "ft_2", "ft_3"]] = [12.0, 16.0, 3.0]
    transformed.loc[1, ["ft_1", "ft_2", "ft_3"]] = [1.0, 1.0, 1.0]
    transformed.insert(1, "gdpr", [1, 0])
    transformed.insert(2, "cluster", [1, 1])

    def evaluate_scenario(scenario):
        return evaluate(
            [[0.0, 0.0], [1.0, 1.0]],
            [1, 1],
            original,
            assignments,
            {"gdpr_20": scenario},
            {1: 2},
            {"gdpr_20": {"requested": 0.2, "overall": 0.2, "train": 0.2, "test": 0.2}},
            small_config,
        )["scenarios"]["gdpr_20"]

    result = evaluate_scenario(transformed)
    assert result["replaced_rows_mae"] == pytest.approx(3.0)
    assert result["replaced_rows_mape"] == pytest.approx(20.0)
    assert result["replaced_rows_mape_status"] == "available"
    assert result["replaced_rows_mape_valid_cells"] == 2
    assert result["replaced_rows_mape_excluded_zero_cells"] == 1

    changed_nonreplaced = transformed.copy()
    changed_nonreplaced.loc[1, ["ft_1", "ft_2", "ft_3"]] = [
        1000.0,
        -2000.0,
        3000.0,
    ]
    changed_result = evaluate_scenario(changed_nonreplaced)
    assert changed_result["replaced_rows_mae"] == pytest.approx(
        result["replaced_rows_mae"]
    )
    assert changed_result["replaced_rows_mape"] == pytest.approx(
        result["replaced_rows_mape"]
    )
    assert changed_result["overall_test_mae"] != result["overall_test_mae"]

    small_config["nonconsent"].update(
        {"start_percent": 20, "end_percent": 20, "step_percent": 10}
    )
    markdown_metrics = {
        "n_clusters": 3,
        "average_training_cluster_size": 2,
        "timings_seconds": {"fitting": 1.0},
        "scenarios": {"gdpr_20": result},
    }
    reporting.export_overall_comparison(small_config, markdown_metrics)
    original_row = (
        small_config.artifact_dir / "tables/overall_comparison.md"
    ).read_text(encoding="utf-8").splitlines()[4]
    markdown_metrics["scenarios"]["gdpr_20"] = changed_result
    reporting.export_overall_comparison(small_config, markdown_metrics)
    changed_row = (
        small_config.artifact_dir / "tables/overall_comparison.md"
    ).read_text(encoding="utf-8").splitlines()[4]
    assert changed_row == original_row

    all_zero_original = original.copy()
    all_zero_original.loc[0, ["ft_1", "ft_2", "ft_3"]] = 0.0
    all_zero_transformed = transformed.copy()
    all_zero_transformed.loc[0, ["ft_1", "ft_2", "ft_3"]] = [1.0, 2.0, 3.0]
    zero_metrics = evaluate(
        [[0.0, 0.0], [1.0, 1.0]],
        [1, 1],
        all_zero_original,
        assignments,
        {"gdpr_20": all_zero_transformed},
        {1: 2},
        {"gdpr_20": {"requested": 0.2, "overall": 0.2, "train": 0.2, "test": 0.2}},
        small_config,
    )["scenarios"]["gdpr_20"]
    assert zero_metrics["replaced_rows_mae"] == pytest.approx(2.0)
    assert zero_metrics["replaced_rows_mape"] is None
    assert zero_metrics["replaced_rows_mape_status"] == (
        "unavailable_no_nonzero_original_values"
    )
    assert zero_metrics["replaced_rows_mape_valid_cells"] == 0
    assert zero_metrics["replaced_rows_mape_excluded_zero_cells"] == 3
