"""Configured full-pipeline orchestration."""

from typing import List

from .config import LoadedConfig
from . import stages


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
    """Execute the validated stage sequence and return concise stage results."""
    if not isinstance(config, LoadedConfig):
        raise TypeError("run() requires a config returned by load_config()")
    results = []
    for step in config["pipeline"]["steps"]:
        results.append(_FUNCTIONS[step](config))
    return results
