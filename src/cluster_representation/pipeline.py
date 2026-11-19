"""Configured full-pipeline orchestration."""

from typing import List

from .config import LoadedConfig
from . import stages
from . import reporting


_FUNCTIONS = {
    "generate_dataset": stages.generate_dataset,
    "split_dataset": stages.split_dataset,
    "fit_preprocessor": stages.fit_preprocessor,
    "train_model": stages.train_model,
    "build_prototypes": stages.build_prototypes,
    "transform_test": stages.transform_test,
    "evaluate": stages.evaluate,
}


def run(config: LoadedConfig) -> List[stages.StageResult]:
    """Run shared data stages once and each selected model in configured order."""
    if not isinstance(config, LoadedConfig):
        raise TypeError("run() requires a config returned by load_config()")
    results = []
    shared_steps = {"generate_dataset", "split_dataset", "fit_preprocessor"}
    steps = config["pipeline"]["steps"]
    first_model_config = config.for_model(config.models_run[0])
    for step in steps:
        if step in shared_steps:
            results.append(_FUNCTIONS[step](first_model_config))

    overall_reports = []
    cluster_selection_reports = []
    feature_reports = []
    for model_name in config.models_run:
        model_config = config.for_model(model_name)
        for step in steps:
            if step in shared_steps:
                continue
            result = _FUNCTIONS[step](model_config)
            results.append(result)
            if step == "evaluate":
                overall_path = result.artifacts.get("overall_comparison")
                if overall_path:
                    overall_reports.append(
                        (config.artifact_dir / overall_path).read_text(
                            encoding="utf-8"
                        )
                    )
                selection_path = result.artifacts.get(
                    "cluster_selection_comparison"
                )
                if selection_path:
                    cluster_selection_reports.append(
                        (config.artifact_dir / selection_path).read_text(
                            encoding="utf-8"
                        )
                    )
                feature_path = result.artifacts.get("feature_evaluation")
                if feature_path:
                    feature_reports.append(
                        (config.artifact_dir / feature_path).read_text(
                            encoding="utf-8"
                        )
                    )
    reporting.export_multi_model_comparisons(
        config, overall_reports, cluster_selection_reports, feature_reports
    )
    return results
