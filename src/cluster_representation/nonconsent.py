"""Generation of independent or nested non-consent scenarios."""

from typing import Any, List, Mapping

import numpy as np
import pandas as pd

from .randomness import generator


def scenario_percentages(config: Mapping[str, Any]) -> List[int]:
    settings = config["nonconsent"]
    return list(
        range(
            settings["start_percent"],
            settings["end_percent"] + 1,
            settings["step_percent"],
        )
    )


def generate_nonconsent(config: Mapping[str, Any]) -> pd.DataFrame:
    """Create binary scenario columns, preserving IDs and scenario stability."""
    settings = config["nonconsent"]
    n_rows = int(config["dataset"]["n_rows"])
    ids = np.arange(1, n_rows + 1, dtype=np.int64)
    percentages = scenario_percentages(config)
    columns = {config["dataset"]["id_column"]: ids}

    if settings["assignment"] == "nested":
        order = generator(config["dataset"]["seed"], "nonconsent:nested").permutation(n_rows)
        for percent in percentages:
            count = n_rows * percent // 100
            values = np.zeros(n_rows, dtype=np.int8)
            values[order[:count]] = 1
            columns["{}_{}".format(settings["prefix"], percent)] = values
    elif settings["assignment"] == "independent":
        for percent in percentages:
            count = n_rows * percent // 100
            order = generator(
                config["dataset"]["seed"], "nonconsent:{}".format(percent)
            ).choice(n_rows, size=count, replace=False)
            values = np.zeros(n_rows, dtype=np.int8)
            values[order] = 1
            columns["{}_{}".format(settings["prefix"], percent)] = values
    else:
        raise ValueError("Unsupported non-consent assignment: {}".format(settings["assignment"]))
    return pd.DataFrame(columns)
