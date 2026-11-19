"""Package-owned tables and plots for notebook and CLI display."""

import math
import json
import re
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from . import nonconsent, storage, transformation
from .config import LoadedConfig


_COMBINED_COLUMNS = [
    "model",
    "n_clusters",
    "constraint_applicability",
    "nonconsent_percent",
    "smallest_cluster_size",
    "silhouette_score",
    "davies_bouldin_index",
    "calinski_harabasz_index",
    "mae",
    "mae_replaced_rows",
    "realized_test_nonconsent_percent",
]

_OVERALL_COMPARISON_PATH = "tables/overall_comparison.md"
_FEATURE_EVALUATION_PATH = "tables/feature_evaluation.md"
_CLUSTER_SELECTION_COMPARISON_PATH = "tables/cluster_selection_comparison.md"


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
                "replaced_rows_mae": values.get("replaced_rows_mae"),
                "replaced_rows_mape": values.get("replaced_rows_mape"),
                "replaced_rows_nmae": values.get("replaced_rows_nmae"),
                "replaced_rows_nmae_status": values.get(
                    "replaced_rows_nmae_status"
                ),
                "per_feature_replaced_rows_mae": values.get(
                    "per_feature_replaced_rows_mae"
                ),
                "per_feature_replaced_rows_mae_status": values.get(
                    "per_feature_replaced_rows_mae_status"
                ),
                "per_feature_replaced_rows_mape": values.get(
                    "per_feature_replaced_rows_mape"
                ),
                "per_feature_replaced_rows_mape_status": values.get(
                    "per_feature_replaced_rows_mape_status"
                ),
                "per_feature_replaced_rows_mape_valid_cells": values.get(
                    "per_feature_replaced_rows_mape_valid_cells"
                ),
                "per_feature_replaced_rows_mape_excluded_zero_cells": values.get(
                    "per_feature_replaced_rows_mape_excluded_zero_cells"
                ),
                "per_feature_replaced_rows_nmae": values.get(
                    "per_feature_replaced_rows_nmae"
                ),
                "per_feature_replaced_rows_nmae_status": values.get(
                    "per_feature_replaced_rows_nmae_status"
                ),
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
            "constraint_applicability": "Constraint applicability",
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


def export_overall_comparison(
    config: LoadedConfig, metrics: Mapping[str, Any]
) -> str:
    """Write each configured non-consent scenario as a row in Markdown."""
    if "n_clusters" not in metrics:
        raise ValueError("Overall comparison metrics are missing n_clusters")

    model_name = config.model_config["name"]
    cluster_count = metrics["n_clusters"]
    clustering_time = metrics.get("timings_seconds", {}).get("fitting")
    if clustering_time is None and model_name == "constrained_kmeans":
        clustering_time = metrics.get("timings_seconds", {}).get("training")
    scenario_results = metrics.get("scenarios", {})
    scenarios = []
    prefix = config["nonconsent"]["prefix"]
    for percent in nonconsent.scenario_percentages(config):
        scenario_name = "{}_{}".format(prefix, percent)
        if scenario_name not in scenario_results:
            raise ValueError(
                "Overall comparison is missing calculated results for configured "
                "scenario {!r}".format(scenario_name)
            )
        scenarios.append((percent, scenario_results[scenario_name]))
    scenarios.sort(key=lambda item: item[0])

    method = "{} (k={})".format(model_name, cluster_count)
    rows = []
    for percent, scenario_metrics in scenarios:
        values = [
            method,
            str(percent),
            _format_available_metric(
                metrics.get("classification_metrics", {}).get("average_precision")
            ),
            _format_available_metric(
                metrics.get("classification_metrics", {}).get("f1")
            ),
            _markdown_number(scenario_metrics.get("replaced_rows_nmae")),
            _markdown_number(scenario_metrics.get("replaced_rows_mape")),
            _markdown_number(cluster_count),
            _markdown_number(metrics.get("average_training_cluster_size")),
            _markdown_number(clustering_time),
        ]
        rows.append("| {} |".format(" | ".join(values)))
    constraint_note = (
        "Constraints: not applicable (unconstrained model)."
        if metrics.get("constraint_applicability") == "not applicable"
        else None
    )
    content = "\n".join(
        [
            "### Overall comparison of evaluated methods",
            "",
            "| Method | NC (%) | AP | F1 | N MAE | MAPE (%) | \\|G\\| | k̄ | Time (s) |",
            "|--------|--------|----|----|-------|----------|-------|----|----------|",
            *rows,
            "",
            "N MAE and MAPE use replaced rows only. N MAE is the",
            "equal-weight mean of feature MAEs divided by their training",
            "standard deviations (ddof=0). Lower is better. N MAE is",
            "measured in training standard deviations, not percentages.",
            "Constant training features are excluded from N MAE and",
            "reported separately. MAPE excludes zero original values.",
            *([constraint_note] if constraint_note else []),
            "",
        ]
    )
    _write_report_with_journal(config, _OVERALL_COMPARISON_PATH, content, overwrite_existing=True)
    return _OVERALL_COMPARISON_PATH


def export_feature_evaluation(
    config: LoadedConfig, metrics: Mapping[str, Any]
) -> str:
    """Write saved replaced-row metrics at the configured feature granularity."""
    model_name = config.model_config["name"]
    if "n_clusters" not in metrics:
        raise ValueError("Feature evaluation metrics are missing n_clusters")
    method = "{} (k={})".format(model_name, metrics["n_clusters"])
    scenarios = metrics.get("scenarios", {})
    feature_stds = (
        metrics.get("training_feature_scales", {}).get("feature_stds", {})
    )
    rows = []
    missing_per_feature_metrics = False
    for percent in nonconsent.scenario_percentages(config):
        scenario_name = "{}_{}".format(config["nonconsent"]["prefix"], percent)
        scenario = scenarios.get(scenario_name, {})
        per_feature_mae = scenario.get("per_feature_replaced_rows_mae")
        per_feature_mape = scenario.get("per_feature_replaced_rows_mape")
        per_feature_nmae = scenario.get("per_feature_replaced_rows_nmae")
        valid_cells = scenario.get("per_feature_replaced_rows_mape_valid_cells")
        excluded_zero_cells = scenario.get(
            "per_feature_replaced_rows_mape_excluded_zero_cells"
        )
        if not all(
            isinstance(value, Mapping)
            for value in (
                per_feature_mae,
                per_feature_mape,
                per_feature_nmae,
                valid_cells,
                excluded_zero_cells,
            )
        ):
            missing_per_feature_metrics = True
        per_feature_mae = per_feature_mae or {}
        per_feature_mape = per_feature_mape or {}
        per_feature_nmae = per_feature_nmae or {}
        valid_cells = valid_cells or {}
        excluded_zero_cells = excluded_zero_cells or {}
        for feature in config["features"]:
            feature_name = feature["name"]
            values = [
                method,
                str(percent),
                feature_name,
                _markdown_number(scenario.get("n_replaced")),
                _markdown_number(per_feature_mae.get(feature_name)),
                _markdown_number(per_feature_mape.get(feature_name)),
                _markdown_number(per_feature_nmae.get(feature_name)),
                _markdown_number(feature_stds.get(feature_name)),
                _markdown_number(valid_cells.get(feature_name)),
                _markdown_number(excluded_zero_cells.get(feature_name)),
            ]
            rows.append("| {} |".format(" | ".join(values)))
    content = [
        "### Feature-level evaluation",
        "",
        "| Method | NC (%) | Feature | Replaced rows | MAE | MAPE (%) | N MAE | Training SD | MAPE valid cells | MAPE excluded zero cells |",
        "|--------|--------|---------|---------------|-----|----------|-------|-------------|------------------|--------------------------|",
        *rows,
        "",
        "All feature metrics use replaced rows only.",
        "Lower MAE, MAPE and N MAE are better.",
        "MAE uses original feature units.",
        "MAPE excludes zero original values.",
        "N MAE uses training SD (ddof=0) and is not a percentage.",
        "Constant training features are excluded from N MAE.",
    ]
    if missing_per_feature_metrics:
        content.append(
            "Some saved results lack per-feature metrics; those values are "
            "unavailable (—) because the results predate per-feature evaluation."
        )
    content.append("")
    _write_report_with_journal(
        config,
        _FEATURE_EVALUATION_PATH,
        "\n".join(content),
        overwrite_existing=True,
    )
    return _FEATURE_EVALUATION_PATH


def export_multi_model_comparisons(
    config: LoadedConfig,
    overall_reports: List[str],
    cluster_selection_reports: List[str],
    feature_reports: Optional[List[str]] = None,
) -> Dict[str, str]:
    """Combine the per-model Markdown tables produced during this run."""
    outputs = {}
    if overall_reports:
        content = _merge_model_markdown_tables(
            overall_reports,
            "| Method | NC (%) |",
            "### Overall comparison of evaluated methods",
        )
        _write_report_with_journal(
            config,
            _OVERALL_COMPARISON_PATH,
            content,
            overwrite_existing=True,
        )
        outputs["overall_comparison"] = _OVERALL_COMPARISON_PATH
    if cluster_selection_reports:
        content = _merge_model_markdown_tables(
            cluster_selection_reports,
            "| Method | Requested k |",
            "### Cluster selection comparison",
        )
        _write_report_with_journal(
            config,
            _CLUSTER_SELECTION_COMPARISON_PATH,
            content,
            overwrite_existing=True,
        )
        outputs["cluster_selection_comparison"] = _CLUSTER_SELECTION_COMPARISON_PATH
    if feature_reports:
        content = _merge_model_markdown_tables(
            feature_reports,
            "| Method | NC (%) | Feature |",
            "### Feature-level evaluation",
        )
        _write_report_with_journal(
            config,
            _FEATURE_EVALUATION_PATH,
            content,
            overwrite_existing=True,
        )
        outputs["feature_evaluation"] = _FEATURE_EVALUATION_PATH
    outputs.update({"journal_" + key: _journal_report_path(path)
                    for key, path in list(outputs.items())})
    return outputs


def export_saved_evaluation_reports(config: LoadedConfig) -> Dict[str, str]:
    """Regenerate comparison reports from saved per-model metrics without fitting."""
    overall_reports = []
    feature_reports = []
    cluster_selection_reports = []
    for model_name in config.models_run:
        model_config = config.for_model(model_name)
        selected_k = storage.selected_candidate(model_config)
        metrics_path = (
            "metrics/{}/k_{}/results.json".format(model_name, selected_k)
            if selected_k is not None
            else "metrics/results.json"
        )
        metrics = storage.read_json(model_config, metrics_path)
        overall_path = export_overall_comparison(model_config, metrics)
        feature_path = export_feature_evaluation(model_config, metrics)
        overall_reports.append(
            (config.artifact_dir / overall_path).read_text(encoding="utf-8")
        )
        feature_reports.append(
            (config.artifact_dir / feature_path).read_text(encoding="utf-8")
        )
        if model_config.model_config.get("cluster_search", {}).get("enabled", False):
            selection_path = export_cluster_selection_comparison(
                model_config, metrics
            )
            cluster_selection_reports.append(
                (config.artifact_dir / selection_path).read_text(encoding="utf-8")
            )
    return export_multi_model_comparisons(
        config,
        overall_reports,
        cluster_selection_reports,
        feature_reports,
    )


def export_feature_evaluation_from_saved_metrics(
    config: LoadedConfig, metrics: Mapping[str, Any]
) -> str:
    """Public convenience wrapper for regenerating a feature report from metrics."""
    return export_feature_evaluation(config, metrics)


def _merge_model_markdown_tables(
    documents: List[str], header_prefix: str, title: str
) -> str:
    headers = None
    table_rows = []
    footnotes = []
    for document in documents:
        lines = document.splitlines()
        header_index = next(
            (
                index
                for index, line in enumerate(lines)
                if line.startswith(header_prefix)
            ),
            None,
        )
        if header_index is None or header_index + 1 >= len(lines):
            raise ValueError("A model comparison report has no recognized table")
        current_header = lines[header_index : header_index + 2]
        if headers is None:
            headers = current_header
        elif current_header != headers:
            raise ValueError("Per-model comparison report columns do not match")
        index = header_index + 2
        while index < len(lines) and lines[index].startswith("|"):
            table_rows.append(lines[index])
            index += 1
        footnotes.extend(line for line in lines[index:] if line.strip())

    return "\n".join(
        [
            title,
            "",
            *(headers or []),
            *table_rows,
            "",
            *dict.fromkeys(footnotes),
            "",
        ]
    )


def export_cluster_selection_comparison(
    config: LoadedConfig, evaluation_metrics: Mapping[str, Any]
) -> str:
    """Write saved training metrics and model-selection results."""
    selection = _read_cluster_selection(config)
    records = _cluster_selection_records(config, evaluation_metrics, selection)
    if not records:
        raise ValueError("No successfully trained model configurations to compare")

    configuration_groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for record in records:
        group_key = (record["method"], record["configuration_key"])
        configuration_groups.setdefault(group_key, []).append(record)

    group_counts: Dict[str, int] = {}
    for method, _ in configuration_groups:
        group_counts[method] = group_counts.get(method, 0) + 1
    for record in records:
        if group_counts[record["method"]] > 1:
            record["display_method"] = "{} [{}]".format(
                record["method"], record["configuration_label"]
            )
        else:
            record["display_method"] = record["method"]

    _mark_metric_winners(records, "silhouette", maximize=True)
    _mark_metric_winners(records, "davies_bouldin", maximize=False)
    _mark_metric_winners(records, "calinski_harabasz", maximize=True)
    records.sort(
        key=lambda record: (
            record["method"],
            record["configuration_key"],
            record["requested_k"],
        )
    )

    rows = []
    for record in records:
        values = [
            record["display_method"],
            str(record["requested_k"]),
            _markdown_number(record["actual_clusters"]),
            _format_ranked_metric(record, "silhouette"),
            _format_ranked_metric(record, "davies_bouldin"),
            _format_ranked_metric(record, "calinski_harabasz"),
            (
                _markdown_number(record.get("selection_score"))
                if selection["selection_mode"] == "auto"
                else "—"
            ),
            "Yes" if record["requested_k"] == selection["selected_n_clusters"] else "No",
            _markdown_number(record["training_time_seconds"]),
        ]
        rows.append("| {} |".format(" | ".join(values)))
    content = "\n".join(
        [
            "### Cluster selection comparison",
            "",
            "| Method | Requested k | Actual clusters | Silhouette ↑ | Davies–Bouldin ↓ | Calinski–Harabasz ↑ | Selection score ↓ | Selected | Training time (s) |",
            "|--------|-------------|-----------------|--------------|-----------------|--------------------|-------------------|----------|------------------|",
            *rows,
            "",
            "Bold values identify the best result for each metric within each method and equivalent configuration group.",
            "Higher silhouette and Calinski–Harabasz are better.",
            "Lower Davies–Bouldin is better.",
            "Metrics may favor different numbers of clusters.",
            (
                "Automatic selection averages the three average metric ranks "
                "(silhouette and Calinski–Harabasz descending; Davies–Bouldin "
                "ascending). Lowest score wins; ties select the smaller requested k."
            ),
            "Selection mode: {}. Configured value: {!r}; resolved k: {}.".format(
                selection["selection_mode"],
                selection["configured_selected_n_clusters"],
                selection["selected_n_clusters"],
            ),
            "Constraint applicability: {}.".format(
                selection.get("constraint_applicability", "applicable")
            ),
            *(
                [
                    "Excluded candidates: "
                    + "; ".join(
                        "k={}: {}".format(
                            item["requested_n_clusters"],
                            ", ".join(item["reasons"]),
                        )
                        for item in selection["excluded_candidates"]
                    )
                ]
                if selection["excluded_candidates"]
                else []
            ),
            "",
        ]
    )
    _write_report_with_journal(
        config,
        _CLUSTER_SELECTION_COMPARISON_PATH,
        content,
        overwrite_existing=True,
    )
    return _CLUSTER_SELECTION_COMPARISON_PATH


def _read_cluster_selection(config: LoadedConfig) -> Dict[str, Any]:
    if config.model_config.get("cluster_search", {}).get("enabled", False):
        fingerprint = storage.selection_fingerprint(config)
        path = storage.selection_artifact_path(config)
        storage.require_stage(
            config,
            storage.selection_stage_key(config),
            fingerprint,
            [path],
        )
        return storage.read_json(config, path)

    configured_k = int(config.model_config["parameters"]["n_clusters"])
    return {
        "selection_mode": "fixed",
        "configured_selected_n_clusters": configured_k,
        "selected_n_clusters": configured_k,
        "candidate_results": [],
        "excluded_candidates": [],
        "constraint_applicability": (
            "applicable"
            if config.model_config["name"] == "constrained_kmeans"
            else "not applicable"
        ),
    }


def _cluster_selection_records(
    config: LoadedConfig,
    evaluation_metrics: Mapping[str, Any],
    selection: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    model_name = config.model_config["name"]
    if config.model_config.get("cluster_search", {}).get("enabled", False):
        comparison_path = _search_comparison_path(config)
        search_fingerprint = storage.cluster_search_fingerprint(config)
        storage.require_stage(
            config,
            _search_stage_key(config),
            search_fingerprint,
            [comparison_path],
        )
        comparison = storage.read_frame(config, comparison_path)
        comparison_by_k = {
            int(row["n_clusters"]): row
            for row in comparison.to_dict(orient="records")
        }
        records = []
        candidates = selection.get("candidate_results", [])
        if len(candidates) != len(comparison_by_k):
            raise ValueError(
                "Saved selection metadata does not cover every cluster-search candidate"
            )
        for candidate in candidates:
            requested_k = int(candidate["requested_n_clusters"])
            comparison_row = comparison_by_k.get(requested_k)
            if comparison_row is None:
                raise ValueError(
                    "Saved selection metadata references unknown candidate k={}".format(
                        requested_k
                    )
                )
            metadata = {}
            if comparison_row.get("status") == "success":
                metadata_path = "models/{}/k_{}/metadata.json".format(
                    model_name, requested_k
                )
                train_key = storage.candidate_stage_key(
                    "train_model", requested_k, model_name
                )
                storage.require_stage(
                    config,
                    train_key,
                    storage.candidate_training_fingerprint(config, requested_k),
                    [metadata_path],
                )
                metadata = storage.read_json(config, metadata_path)
                if metadata.get("status") != "success":
                    raise ValueError(
                        "Candidate comparison status does not match saved metadata for "
                        "k={}".format(requested_k)
                    )
                parameters = _metadata_mapping(metadata.get("model_parameters"))
                if int(parameters.get("n_clusters", -1)) != requested_k:
                    raise ValueError(
                        "Candidate comparison requested k does not match saved model "
                        "metadata for k={}".format(requested_k)
                    )
            records.append(
                _cluster_selection_record_from_candidate(
                    model_name, candidate, comparison_row, metadata
                )
            )
        return records

    metadata = storage.read_json(config, "models/metadata.json")
    parameters = metadata.get("parameters", {})
    constraints = metadata.get("constraints", {})
    requested_k = parameters.get("n_clusters")
    if requested_k is None:
        raise ValueError("Saved model metadata is missing requested n_clusters")
    return [
        _cluster_selection_record(
            model_name,
            int(requested_k),
            metadata,
            evaluation_metrics,
            parameters,
            constraints,
        )
    ]


def _cluster_selection_record_from_candidate(
    method: str,
    candidate: Mapping[str, Any],
    comparison: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> Dict[str, Any]:
    requested_k = int(candidate["requested_n_clusters"])
    configuration = candidate.get("configuration", {})
    configuration_key = candidate.get("configuration_key") or json.dumps(
        configuration, sort_keys=True, default=str
    )
    return {
        "method": method,
        "requested_k": requested_k,
        "actual_clusters": metadata.get(
            "n_clusters", candidate.get("actual_clusters")
        ),
        "training_time_seconds": metadata.get(
            "training_time_seconds", candidate.get("training_time_seconds")
        ),
        "silhouette": candidate["metrics"].get("silhouette"),
        "davies_bouldin": candidate["metrics"].get("davies_bouldin"),
        "calinski_harabasz": candidate["metrics"].get("calinski_harabasz"),
        "selection_score": candidate.get("selection_score"),
        "configuration_key": configuration_key,
        "configuration_label": _configuration_label(configuration),
        "status": comparison.get("status"),
        "exclusion_reasons": candidate.get("exclusion_reasons", []),
    }


def _cluster_selection_record(
    method: str,
    requested_k: int,
    metadata: Mapping[str, Any],
    evaluation_metrics: Optional[Mapping[str, Any]],
    parameters: Mapping[str, Any],
    constraints: Mapping[str, Any],
) -> Dict[str, Any]:
    actual_clusters = metadata.get("n_clusters")
    if actual_clusters is None and evaluation_metrics is not None:
        actual_clusters = evaluation_metrics.get("n_clusters")
    configuration = {
        "parameters": {
            key: value for key, value in parameters.items() if key != "n_clusters"
        },
        "constraints": dict(constraints),
    }
    configuration_key = json.dumps(configuration, sort_keys=True, default=str)
    configuration_label = _configuration_label(configuration)
    source = evaluation_metrics if evaluation_metrics is not None else metadata
    return {
        "method": method,
        "requested_k": int(requested_k),
        "actual_clusters": actual_clusters,
        "training_time_seconds": metadata.get("training_time_seconds"),
        "silhouette": _saved_metric(source, "silhouette", "silhouette_score"),
        "davies_bouldin": _saved_metric(
            source, "davies_bouldin", "davies_bouldin_index"
        ),
        "calinski_harabasz": _saved_metric(
            source, "calinski_harabasz", "calinski_harabasz_index"
        ),
        "configuration_key": configuration_key,
        "configuration_label": configuration_label,
    }


def _metadata_mapping(value: Any) -> Dict[str, Any]:
    if isinstance(value, str):
        parsed = json.loads(value)
        if not isinstance(parsed, dict):
            raise ValueError("Saved model configuration metadata must be a mapping")
        return parsed
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("Saved model configuration metadata must be a mapping")
    return dict(value)


def _configuration_label(configuration: Mapping[str, Any]) -> str:
    parts = []
    for section in ("parameters", "constraints"):
        settings = configuration[section]
        if settings:
            items = ", ".join(
                "{}={}".format(key, value)
                for key, value in sorted(settings.items())
            )
            parts.append("{}: {}".format(section, items))
    return "; ".join(parts) or "default configuration"


def _saved_metric(
    source: Mapping[str, Any], nested_key: str, flat_key: str
) -> Optional[float]:
    clustering = source.get("clustering_metrics", {})
    metric = clustering.get(nested_key) if isinstance(clustering, Mapping) else None
    if isinstance(metric, Mapping):
        if metric.get("status") != "available":
            return None
        value = metric.get("value")
    else:
        status_keys = {
            "silhouette": "silhouette_status",
            "davies_bouldin": "davies_bouldin_status",
            "calinski_harabasz": "calinski_harabasz_status",
        }
        status = source.get(status_keys[nested_key])
        value = source.get(flat_key)
        if status != "available":
            return None
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return None
    return numeric_value if math.isfinite(numeric_value) else None


def _mark_metric_winners(
    records: List[Dict[str, Any]], metric_name: str, maximize: bool
) -> None:
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for record in records:
        groups.setdefault(
            (record["method"], record["configuration_key"]), []
        ).append(record)
    for group in groups.values():
        available = [
            record[metric_name] for record in group if record[metric_name] is not None
        ]
        if not available:
            continue
        best = max(available) if maximize else min(available)
        for record in group:
            record["{}_winner".format(metric_name)] = (
                record[metric_name] is not None and record[metric_name] == best
            )


def _format_ranked_metric(record: Mapping[str, Any], metric_name: str) -> str:
    value = record[metric_name]
    if value is None:
        return "—"
    formatted = _markdown_number(value)
    if record.get("{}_winner".format(metric_name), False):
        return "**{}**".format(formatted)
    return formatted


def _markdown_number(value: Any) -> str:
    if value is None:
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(number):
        return "—"
    if number.is_integer():
        return str(int(number))
    return str(number)


def _format_available_metric(metric: Any) -> str:
    if not isinstance(metric, Mapping) or metric.get("status") != "available":
        return "—"
    return _markdown_number(metric.get("value"))


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
    model_name = config.model_config["name"]
    constraints_applicable = model_name == "constrained_kmeans"
    minimum_size = (
        config.model_config["constraints"]["min_cluster_size"]
        if constraints_applicable
        else None
    )
    rows: List[Dict[str, Any]] = []

    for candidate in comparison.to_dict(orient="records"):
        if candidate["status"] != "success":
            continue
        n_clusters = int(candidate["n_clusters"])
        model_path = "models/{}/k_{}/model.joblib".format(model_name, n_clusters)
        assignments_path = (
            "models/{}/k_{}/train_cluster_assignments.parquet".format(
                model_name, n_clusters
            )
        )
        metadata_path = "models/{}/k_{}/metadata.json".format(
            model_name,
            n_clusters
        )
        training_key = storage.candidate_stage_key(
            "train_model", n_clusters, model_name
        )
        storage.require_stage(
            config,
            training_key,
            storage.candidate_training_fingerprint(config, n_clusters),
            [model_path, assignments_path, metadata_path],
        )
        prototypes_path = (
            "models/{}/k_{}/cluster_prototypes.parquet".format(
                model_name, n_clusters
            )
        )
        storage.require_stage(
            config,
            storage.candidate_stage_key(
                "build_prototypes", n_clusters, model_name
            ),
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
        if constraints_applicable and smallest_cluster_size < minimum_size:
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
                    "constraint_applicability": candidate.get(
                        "constraint_applicability",
                        "applicable" if constraints_applicable else "not applicable",
                    ),
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
    model_name = config.model_config["name"]
    metrics_path = "metrics/results.json"
    storage.require_stage(
        config,
        "evaluate:{}".format(model_name),
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
                "model": model_name,
                "n_clusters": n_clusters,
                "constraint_applicability": (
                    metrics.get("constraint_applicability", "not applicable")
                ),
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
    scenarios = []
    percentages = nonconsent.scenario_percentages(config)
    for column, percent in zip(nonconsent.scenario_columns(config), percentages):
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
    return "models/{}/search_{}/comparison.parquet".format(
        config.model_config["name"], fingerprint[:12]
    )


def _search_stage_key(config: LoadedConfig) -> str:
    fingerprint = storage.cluster_search_fingerprint(config)
    return "train_model:search:{}".format(fingerprint[:12])


def _write_report_with_journal(
    config: LoadedConfig, path: str, content: str, overwrite_existing: bool = True
) -> None:
    """Write a Markdown report and its copy-ready LaTeX table."""
    storage.write_text(config, path, content, overwrite_existing=overwrite_existing)
    journal_path = _journal_report_path(path)
    storage.write_text(
        config, journal_path, _journal_table(content),
        overwrite_existing=overwrite_existing,
    )


def _journal_report_path(path: str) -> str:
    name = path.rsplit('/', 1)[-1].removesuffix('.md')
    if name == 'cluster_selection_comparison':
        name = 'cluster_selection_compirson'
    return 'tables/journal_{}.tex'.format(name)


def _journal_table(content: str) -> str:
    """Render the report's table, preserving configuration labels and winners."""
    replacements = {
        '\\': r'\textbackslash{}', '&': r'\&', '%': r'\%', '$': r'\$',
        '#': r'\#', '_': r'\_', '{': r'\{', '}': r'\}',
        '~': r'\textasciitilde{}', '^': r'\textasciicircum{}',
        '↑': r'$\uparrow$', '↓': r'$\downarrow$', '—': '--',
        '–': '--', 'k̄': r'$\bar{k}$',
    }

    def cell(value: str) -> str:
        value = value.strip().replace(r'\|', '|')
        bold = value.startswith('**') and value.endswith('**')
        if bold:
            value = value[2:-2]
        try:
            number = float(value)
        except ValueError:
            pass
        else:
            if math.isfinite(number) and ('.' in value or 'e' in value.lower()):
                value = '{:.3f}'.format(number)
        escaped = re.sub(
            r'k̄|[\\&%$#_{}~^↑↓—–]',
            lambda match: replacements[match.group()], value,
        )
        return r'\textbf{' + escaped + '}' if bold else escaped

    lines = [line for line in content.splitlines() if line.startswith('|')]
    rows = [
        [cell(value) for value in re.split(r'(?<!\\)\|', line.strip())[1:-1]]
        for line in lines
        if not re.fullmatch(r'[| :\-]+', line)
    ]
    if not rows:
        raise ValueError('Report contains no table to export')
    title = next(line.lstrip('# ').strip() for line in content.splitlines() if line.startswith('#'))
    label = re.sub(r'[^a-z0-9]+', '_', title.lower()).strip('_')
    output = [
        r'\begin{table}[ht]', r'\centering',
        r'\caption{' + cell(title) + '}', r'\label{tab:' + label + '}',
        r'\footnotesize', r'\begin{tabular}{l' + 'c' * (len(rows[0]) - 1) + '}',
        r'\hline', ' & '.join(rows[0]) + r' \\', r'\hline',
    ]
    previous_method = None
    for row in rows[1:]:
        if previous_method is not None and row[0] != previous_method:
            output.append(r'\hline')
        output.append(' & '.join(row) + r' \\')
        previous_method = row[0]
    output.extend([r'\hline', r'\end{tabular}', r'\end{table}', ''])
    return '\n'.join(output)
