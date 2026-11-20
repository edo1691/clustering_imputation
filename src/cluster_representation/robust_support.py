"""Configuration migration and candidate identities for robust co-clustering."""

from copy import deepcopy
from typing import NamedTuple

ROBUST_MODELS = {"robust_coclustering", "robust_constrained_coclustering"}


class ClusterPair(NamedTuple):
    g: int
    m: int

    def __str__(self):
        return "g_{}_m_{}".format(self.g, self.m)


def normalize_robust_config(model):
    """Migrate equivalent legacy settings without changing algorithm semantics."""
    result = deepcopy(model)
    parameters = result["parameters"]
    legacy = any(key in parameters for key in (
        "n_row_clusters", "n_column_clusters", "random_state", "max_iter"
    )) or "constraints" in result or "trimming" in result
    aliases = {
        "n_row_clusters": "g", "n_column_clusters": "m",
        "random_state": "seed", "max_iter": "iter_max",
    }
    for old, new in aliases.items():
        if old in parameters:
            value = parameters.pop(old)
            _merge(parameters, new, value, old)
    constraints = result.pop("constraints", {})
    if set(constraints) - {"min_row_cluster_size"}:
        raise ValueError("Unsupported legacy robust constraint keys")
    if "min_row_cluster_size" in constraints:
        _merge(parameters, "row_size_min", constraints["min_row_cluster_size"], "min_row_cluster_size")
    trimming = result.pop("trimming", {})
    if set(trimming) - {"row_fraction", "column_fraction"}:
        raise ValueError("Unsupported legacy robust trimming keys")
    if trimming:
        if set(trimming) != {"row_fraction", "column_fraction"}:
            raise ValueError("Legacy trimming requires both row_fraction and column_fraction")
        if any(value != 0 for value in trimming.values()):
            raise ValueError("Legacy row/column trimming is unsupported by the cell-trimming algorithm")
        _merge(parameters, "trim_lvl", 0.0, "legacy zero row/column trimming")
    if legacy:
        result["source_configuration"] = deepcopy(model)
    return result


def _merge(parameters, key, value, origin):
    if key in parameters and parameters[key] != value:
        raise ValueError("Conflicting {} and {} values".format(origin, key))
    parameters[key] = value


def search_candidates(config):
    search = config.model_config["cluster_search"]
    if config.model_config["name"] not in ROBUST_MODELS:
        return search["candidates"]
    if search.get("mode") == "grid":
        return [ClusterPair(*pair) for pair in search["candidates"]]
    return [ClusterPair(g, config.model_config["parameters"]["m"]) for g in search["candidates"]]


def candidate_from_record(config, record):
    if config.model_config["name"] in ROBUST_MODELS:
        return ClusterPair(int(record["n_clusters"]), int(record["requested_column_clusters"]))
    return int(record["n_clusters"])


def candidate_feasibility(parameters, values):
    """Check deterministic input bounds before invoking the unchanged algorithm."""
    import numpy as np
    from .models.custom.robcoclust_constrained import _check_size_bounds

    g, m = parameters["g"], parameters["m"]
    n, p = values.shape
    if g > n or m > p:
        return "Non-empty blocks require g <= training rows and m <= features (g={}, m={}, rows={}, features={})".format(g, m, n, p)
    row_min = parameters.get("row_size_min")
    row_max = parameters.get("row_size_max")
    col_min = parameters.get("col_size_min")
    col_max = parameters.get("col_size_max")
    try:
        _check_size_bounds(n, g, 1 if row_min is None else row_min, n if row_max is None else row_max, "rows")
        _check_size_bounds(p, m, 1 if col_min is None else col_min, p if col_max is None else col_max, "columns")
    except ValueError as exc:
        return str(exc)
    observed = int((~np.isnan(values)).sum())
    kept = observed - int(np.floor(parameters.get("trim_lvl", 0.10) * observed))
    if g * m * 2 > kept:
        return "Too few observed cells after trimming: {} kept, {} required for g={} m={} (d=1)".format(kept, g * m * 2, g, m)
    return None


def uses_candidate_artifacts(config):
    search = config.model_config.get("cluster_search", {})
    return bool(search.get("enabled") or (
        config.model_config["name"] in ROBUST_MODELS and search.get("mode") == "grid"
    ))
