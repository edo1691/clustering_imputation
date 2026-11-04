"""Synthetic numeric feature generation."""

from typing import Any, Dict, Mapping

import numpy as np
import pandas as pd

from .randomness import generator


def generate_features(config: Mapping[str, Any]) -> pd.DataFrame:
    """Generate an ID column and independent features from their configured distributions.

    Normal uses mean/std; uniform uses inclusive-low/exclusive-high bounds;
    exponential uses scale; Poisson uses lambda; binomial uses trials n and
    success probability p; lognormal uses the mean and sigma of the underlying
    normal distribution.
    """
    dataset = config["dataset"]
    n_rows = int(dataset["n_rows"])
    result: Dict[str, Any] = {
        dataset["id_column"]: np.arange(1, n_rows + 1, dtype=np.int64)
    }
    for feature in config["features"]:
        params = feature["parameters"]
        rng = generator(dataset["seed"], "feature:" + feature["name"])
        distribution = feature["distribution"]
        if distribution == "normal":
            values = rng.normal(params["mean"], params["std"], n_rows)
        elif distribution == "uniform":
            values = rng.uniform(params["low"], params["high"], n_rows)
        elif distribution == "exponential":
            values = rng.exponential(params["scale"], n_rows)
        elif distribution == "poisson":
            values = rng.poisson(params["lam"], n_rows)
        elif distribution == "binomial":
            values = rng.binomial(params["n"], params["p"], n_rows)
        elif distribution == "lognormal":
            values = rng.lognormal(params["mean"], params["sigma"], n_rows)
        else:
            raise ValueError("Unsupported distribution: {}".format(distribution))
        result[feature["name"]] = values
    return pd.DataFrame(result)
