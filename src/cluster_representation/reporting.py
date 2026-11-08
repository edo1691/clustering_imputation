"""Package-owned tables and plots for notebook and CLI display."""

from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from . import storage, transformation
from .config import LoadedConfig


_COMBINED_COLUMNS = [
    "model",
    "n_clusters",
    "nonconsent_percent",
    "smallest_cluster_size",
    "silhouette_score",
    "davies_bouldin_index",
    "calinski_harabasz_index",
    "mae",
    "mae_replaced_rows",
    "realized_test_nonconsent_percent",
]


def metrics_table(metrics: Mapping[str, Any]) -> pd.DataFrame:
    rows = []
    for scenario, values in metrics.get("scenarios", {}).items():
        rows.append(
            {
                "scenario": scenario,
                "requested_nonconsent_rate": values["requested_nonconsent_rate"],
                "realized_overall_nonconsent_rate": values[
                    "realized_overall_nonconsent_rate"
                ],
                "realized_train_nonconsent_rate": values[
                    "realized_train_nonconsent_rate"
                ],
                "realized_test_nonconsent_rate": values[
                    "realized_test_nonconsent_rate"
                ],
                "n_replaced": values["n_replaced"],
                "overall_test_mae": values["overall_test_mae"],
                "replaced_rows_mae": values["replaced_rows_mae"],
            }
        )
    return pd.DataFrame(rows)


def cluster_size_plot(cluster_sizes: Mapping[int, int]) -> Any:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots()
    axis.bar([str(key) for key in cluster_sizes], list(cluster_sizes.values()))
    axis.set(title="Training cluster sizes", xlabel="Cluster", ylabel="Customers")
    figure.tight_layout()
    return figure


def display_report(metrics: Mapping[str, Any]) -> Dict[str, Any]:
    """Return display-ready tables and a cluster-size figure."""
    sizes = {int(key): int(value) for key, value in metrics["training_cluster_sizes"].items()}
    return {
        "scenarios": metrics_table(metrics),
        "training_cluster_sizes": pd.DataFrame(
            [{"cluster": key, "n_clients": value} for key, value in sizes.items()]
        ),
        "cluster_size_plot": cluster_size_plot(sizes),
    }


def cluster_search_report(comparison: pd.DataFrame) -> Dict[str, Any]:
    """Return a cluster-count comparison table and package-owned score plot."""
    required = {
        "n_clusters",
        "silhouette_score",
        "davies_bouldin_index",
        "calinski_harabasz_index",
        "status",
    }
    missing = required - set(comparison.columns)
    if missing:
        raise ValueError(
            "Cluster search comparison is missing columns: {}".format(sorted(missing))
        )
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots()
    successful = comparison.loc[comparison["status"].eq("success")]
    axis.plot(
        successful["n_clusters"],
        successful["silhouette_score"],
        marker="o",
        label="Silhouette",
    )
    axis.set(
        title="Training cluster-count comparison",
        xlabel="Number of clusters",
        ylabel="Silhouette score",
    )
    axis.legend()
    figure.tight_layout()
    return {"comparison": comparison, "score_plot": figure}


def combined_evaluation_report(config: LoadedConfig) -> pd.DataFrame:
    """Export all model/K/scenario results and return a display-ready table."""
    if not isinstance(config, LoadedConfig):
        raise TypeError("Reporting requires a config returned by load_config()")
    if not config["evaluation"]["enabled"]:
        raise ValueError("Combined evaluation reporting is disabled by configuration")

    if config.model_config.get("cluster_search", {}).get("enabled", False):
        rows = _cluster_search_evaluation_rows(config)
    else:
        rows = _single_model_evaluation_rows(config)

    machine_table = pd.DataFrame(rows, columns=_COMBINED_COLUMNS)
    if machine_table.empty:
        raise ValueError("No successfully trained models are available to report")
    machine_table = machine_table.sort_values(
        ["model", "n_clusters", "nonconsent_percent"], kind="stable"
    ).reset_index(drop=True)
    storage.write_csv(
        config, "metrics/combined_evaluation.csv", machine_table, overwrite_existing=True
    )
    storage.write_frame(
        config,
        "metrics/combined_evaluation.parquet",
        machine_table,
        overwrite_existing=True,
    )

    display_table = machine_table.rename(
        columns={
            "model": "Model",
            "n_clusters": "Clusters",
            "nonconsent_percent": "Non-consent (%)",
            "smallest_cluster_size": "Smallest cluster",
            "silhouette_score": "Silhouette ↑",
            "davies_bouldin_index": "Davies–Bouldin ↓",
            "calinski_harabasz_index": "Calinski–Harabasz ↑",
            "mae": "MAE ↓",
            "mae_replaced_rows": "MAE (replaced rows) ↓",
            "realized_test_nonconsent_percent": "Realized test non-consent (%)",
        }
    )
    return display_table.astype(object).where(pd.notna(display_table), "N/A")


def _cluster_search_evaluation_rows(config: LoadedConfig) -> List[Dict[str, Any]]:
    comparison_path = _search_comparison_path(config)
    search_fingerprint = storage.cluster_search_fingerprint(config)
    storage.require_stage(
        config,
        _search_stage_key(config),
        search_fingerprint,
        [comparison_path],
    )

    split_fingerprint = storage.stage_fingerprint(config, "split_dataset")
    storage.require_stage(
        config,
        "split_dataset",
        split_fingerprint,
        [
            "data/train_features.parquet",
            "data/test_features.parquet",
            "data/test_indicators.parquet",
        ],
    )
    preprocessing_fingerprint = storage.stage_fingerprint(config, "fit_preprocessor")
    storage.require_stage(
        config,
        "fit_preprocessor",
        preprocessing_fingerprint,
        ["preprocessing/preprocessor.joblib"],
    )

    comparison = storage.read_frame(config, comparison_path)
    train_features = storage.read_frame(config, "data/train_features.parquet")
    test_features = storage.read_frame(config, "data/test_features.parquet")
    test_indicators = storage.read_frame(config, "data/test_indicators.parquet")
    preprocessor = storage.read_object(config, "preprocessing/preprocessor.joblib")
    feature_names = [feature["name"] for feature in config["features"]]
    id_column = config["dataset"]["id_column"]
    minimum_size = config.model_config["constraints"]["min_cluster_size"]
    rows: List[Dict[str, Any]] = []

    for candidate in comparison.to_dict(orient="records"):
        if candidate["status"] != "success":
            continue
        n_clusters = int(candidate["n_clusters"])
        model_path = "models/constrained_kmeans/k_{}/model.joblib".format(n_clusters)
        assignments_path = (
            "models/constrained_kmeans/k_{}/train_cluster_assignments.parquet".format(
                n_clusters
            )
        )
        metadata_path = "models/constrained_kmeans/k_{}/metadata.json".format(
            n_clusters
        )
        training_key = storage.candidate_stage_key("train_model", n_clusters)
        storage.require_stage(
            config,
            training_key,
            storage.candidate_training_fingerprint(config, n_clusters),
            [model_path, assignments_path, metadata_path],
        )
        prototypes_path = (
            "models/constrained_kmeans/k_{}/cluster_prototypes.parquet".format(
                n_clusters
            )
        )
        storage.require_stage(
            config,
            storage.candidate_stage_key("build_prototypes", n_clusters),
            storage.candidate_prototype_fingerprint(config, n_clusters),
            [prototypes_path],
        )

        model = storage.read_object(config, model_path)
        train_assignments = storage.read_frame(config, assignments_path)
        if train_assignments[id_column].tolist() != train_features[id_column].tolist():
            raise ValueError(
                "Candidate {} training assignments do not align with training rows".format(
                    n_clusters
                )
            )
        fitted_labels = np.asarray(model.labels_, dtype=int)
        if not np.array_equal(
            fitted_labels, train_assignments["cluster"].to_numpy(dtype=int)
        ):
            raise ValueError(
                "Candidate {} assignments differ from its fitted training labels".format(
                    n_clusters
                )
            )
        smallest_cluster_size = int(
            train_assignments["cluster"].value_counts().min()
        )
        if smallest_cluster_size < minimum_size:
            raise ValueError(
                "Candidate {} fitted memberships violate the configured minimum "
                "cluster size".format(n_clusters)
            )

        prototype_frame = storage.read_frame(config, prototypes_path)
        assignments = transformation.assign_test_clusters(
            test_features, config, preprocessor, model
        )["assignments"]
        for scenario_name, percent in _configured_scenarios(config, test_indicators):
            transformed = transformation.transform_test(
                test_features,
                test_indicators[[id_column, scenario_name]],
                prototype_frame,
                assignments,
                config,
                scenario_name,
            )["transformed"]
            errors = np.abs(
                transformed[feature_names].to_numpy(dtype=float)
                - test_features[feature_names].to_numpy(dtype=float)
            )
            replaced = transformed["gdpr"].to_numpy() == config["transformation"][
                "replace_when"
            ]
            rows.append(
                {
                    "model": config.model_config["name"],
                    "n_clusters": n_clusters,
                    "nonconsent_percent": percent,
                    "smallest_cluster_size": smallest_cluster_size,
                    "silhouette_score": candidate["silhouette_score"],
                    "davies_bouldin_index": candidate["davies_bouldin_index"],
                    "calinski_harabasz_index": candidate[
                        "calinski_harabasz_index"
                    ],
                    "mae": float(errors.mean()),
                    "mae_replaced_rows": (
                        float(errors[replaced].mean()) if replaced.any() else None
                    ),
                    "realized_test_nonconsent_percent": float(
                        transformed["gdpr"].mean() * 100.0
                    ),
                }
            )
    return rows


def _single_model_evaluation_rows(config: LoadedConfig) -> List[Dict[str, Any]]:
    metrics_path = "metrics/results.json"
    storage.require_stage(
        config,
        "evaluate",
        storage.stage_fingerprint(config, "evaluate"),
        [metrics_path],
    )
    metrics = storage.read_json(config, metrics_path)
    scenarios = metrics.get("scenarios", {})
    n_clusters = int(metrics["n_clusters"])
    smallest_cluster_size = int(metrics["minimum_training_cluster_size"])
    clustering = metrics["clustering_metrics"]
    rows = []
    for scenario_name, percent in _configured_scenarios(config):
        if scenario_name not in scenarios:
            raise ValueError(
                "Scenario {!r} is missing from saved evaluation metrics".format(
                    scenario_name
                )
            )
        scenario = scenarios[scenario_name]
        rows.append(
            {
                "model": config.model_config["name"],
                "n_clusters": n_clusters,
                "nonconsent_percent": percent,
                "smallest_cluster_size": smallest_cluster_size,
                "silhouette_score": clustering["silhouette"]["value"],
                "davies_bouldin_index": clustering["davies_bouldin"]["value"],
                "calinski_harabasz_index": clustering["calinski_harabasz"][
                    "value"
                ],
                "mae": scenario["overall_test_mae"],
                "mae_replaced_rows": scenario["replaced_rows_mae"],
                "realized_test_nonconsent_percent": (
                    scenario["realized_test_nonconsent_rate"] * 100.0
                ),
            }
        )
    return rows


def _configured_scenarios(
    config: LoadedConfig, indicators: Optional[pd.DataFrame] = None
) -> List[Tuple[str, int]]:
    prefix = config["nonconsent"]["prefix"]
    scenarios = []
    settings = config["nonconsent"]
    for percent in range(
        settings["start_percent"],
        settings["end_percent"] + 1,
        settings["step_percent"],
    ):
        column = "{}_{}".format(prefix, percent)
        if indicators is not None and column not in indicators.columns:
            raise ValueError(
                "Configured scenario {!r} is missing from saved test indicators".format(
                    column
                )
            )
        scenarios.append((column, percent))
    if not scenarios:
        raise ValueError(
            "No configured evaluation scenarios were found"
        )
    return scenarios


def _search_comparison_path(config: LoadedConfig) -> str:
    fingerprint = storage.cluster_search_fingerprint(config)
    return "models/constrained_kmeans/search_{}/comparison.parquet".format(
        fingerprint[:12]
    )


def _search_stage_key(config: LoadedConfig) -> str:
    fingerprint = storage.cluster_search_fingerprint(config)
    return "train_model:search:{}".format(fingerprint[:12])
