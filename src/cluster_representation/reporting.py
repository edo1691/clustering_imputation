"""Package-owned tables and plots for notebook and CLI display."""

from typing import Any, Dict, Mapping

import pandas as pd


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
