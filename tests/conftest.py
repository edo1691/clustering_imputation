import copy

import pytest
import yaml

from cluster_representation import load_config


@pytest.fixture
def small_config_path(tmp_path):
    source = yaml.safe_load(
        open("config/default.yml", "r", encoding="utf-8")
    )
    source = copy.deepcopy(source)
    source["dataset"]["n_rows"] = 200
    source["features"] = [
        {
            "name": "ft_1",
            "distribution": "normal",
            "parameters": {"mean": 0.0, "std": 1.0},
        },
        {
            "name": "ft_2",
            "distribution": "uniform",
            "parameters": {"low": 0.0, "high": 10.0},
        },
        {
            "name": "ft_3",
            "distribution": "poisson",
            "parameters": {"lam": 3},
        },
    ]
    source["nonconsent"].update(
        {"start_percent": 0, "end_percent": 100, "step_percent": 20}
    )
    source["split"]["test_size"] = 0.25
    source["output"]["root"] = "../artifacts"
    source["output"]["run_name"] = "test_run"
    config_dir = tmp_path / "config"
    models_dir = config_dir / "models"
    models_dir.mkdir(parents=True)
    model_config = {
        "name": "constrained_kmeans",
        "parameters": {
            "n_clusters": 3,
            "random_state": 12345,
            "max_iter": 40,
            "n_init": 2,
        },
        "constraints": {
            "type": "minimum_cluster_size",
            "min_cluster_size": 5,
        },
    }
    with (models_dir / "constrained.yml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(model_config, stream)
    source["models"] = {
        "run": ["constrained_kmeans"],
        "configs": {"constrained_kmeans": "models/constrained.yml"},
    }
    path = config_dir / "small.yml"
    with path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(source, stream, sort_keys=False)
    return path


@pytest.fixture
def small_config(small_config_path):
    return load_config(small_config_path)


@pytest.fixture
def small_search_config_path(small_config_path):
    settings = yaml.safe_load(small_config_path.read_text(encoding="utf-8"))
    model_path = small_config_path.parent / "models" / "search.yml"
    model_config = {
        "name": "constrained_kmeans",
        "parameters": {
            "random_state": 12345,
            "max_iter": 40,
            "n_init": 2,
        },
        "cluster_search": {
            "enabled": True,
            "candidates": [2, 3, 20],
            "selected_n_clusters": 3,
        },
        "constraints": {
            "type": "minimum_cluster_size",
            "min_cluster_size": 10,
        },
    }
    model_path.write_text(yaml.safe_dump(model_config), encoding="utf-8")
    settings["models"]["run"] = ["constrained_kmeans"]
    settings["models"]["configs"] = {
        "constrained_kmeans": "models/search.yml"
    }
    settings["output"]["run_name"] = "search_run"
    small_config_path.write_text(yaml.safe_dump(settings), encoding="utf-8")
    return small_config_path


@pytest.fixture
def small_search_config(small_search_config_path):
    return load_config(small_search_config_path)
