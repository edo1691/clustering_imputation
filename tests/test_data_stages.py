import copy

import numpy as np
import pandas as pd
import pytest
import yaml

from cluster_representation.config import ConfigurationError, load_config
from cluster_representation.generation import generate_features
from cluster_representation.nonconsent import generate_nonconsent
from cluster_representation.preprocessing import fit_preprocessor
from cluster_representation.splitting import split_data


def test_config_rejects_unknown_keys_and_invalid_feature_names(small_config_path):
    source = yaml.safe_load(small_config_path.read_text(encoding="utf-8"))
    source["unexpected"] = True
    small_config_path.write_text(yaml.safe_dump(source), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="unknown key"):
        load_config(small_config_path)

    source.pop("unexpected")
    source["features"][1]["name"] = "ft_4"
    small_config_path.write_text(yaml.safe_dump(source), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="contiguous and ordered"):
        load_config(small_config_path)


def test_config_resolves_model_and_output_paths_from_yaml(small_config):
    assert small_config.model_config_path.is_absolute()
    assert small_config.artifact_dir == small_config.source_path.parent.parent / "artifacts" / "test_run"


def test_config_accepts_constrained_coclustering_and_rejects_trimming(
    small_config_path,
):
    source = yaml.safe_load(small_config_path.read_text(encoding="utf-8"))
    source["model"]["config_path"] = "models/constrained_coclustering.yml"
    model_config = {
        "name": "constrained_coclustering",
        "parameters": {
            "n_row_clusters": 3,
            "n_column_clusters": 2,
            "random_state": 12345,
            "max_iter": 100,
            "n_init": 5,
        },
        "constraints": {"min_row_cluster_size": 5},
    }
    model_path = small_config_path.parent / "models/constrained_coclustering.yml"
    model_path.write_text(yaml.safe_dump(model_config), encoding="utf-8")
    small_config_path.write_text(yaml.safe_dump(source), encoding="utf-8")
    loaded = load_config(small_config_path)
    assert loaded.model_config["name"] == "constrained_coclustering"
    assert "trimming" not in loaded.model_config

    model_config["trimming"] = {"row_fraction": 0.1, "column_fraction": 0.1}
    model_path.write_text(yaml.safe_dump(model_config), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="unknown key"):
        load_config(small_config_path)


def test_generation_reproducibility_and_independent_feature_streams(small_config):
    first = generate_features(small_config)
    repeated = generate_features(small_config)
    pd.testing.assert_frame_equal(first, repeated)
    assert first["id_row"].tolist() == list(range(1, 201))

    extended = copy.deepcopy(small_config)
    extended["features"].append(
        {
            "name": "ft_4",
            "distribution": "binomial",
            "parameters": {"n": 4, "p": 0.5},
        }
    )
    pd.testing.assert_frame_equal(
        first,
        generate_features(extended)[first.columns],
    )
    changed_nonconsent = copy.deepcopy(small_config)
    changed_nonconsent["nonconsent"]["assignment"] = "independent"
    pd.testing.assert_frame_equal(first, generate_features(changed_nonconsent))
    assert pd.api.types.is_integer_dtype(first["ft_3"])


def test_nonconsent_counts_nested_membership_and_stable_existing_scenarios(small_config):
    indicators = generate_nonconsent(small_config)
    for percentage in range(0, 101, 20):
        column = "gdpr_{}".format(percentage)
        assert int(indicators[column].sum()) == 200 * percentage // 100
    for lower, upper in zip(range(0, 100, 20), range(20, 101, 20)):
        assert (indicators["gdpr_{}".format(lower)] <= indicators["gdpr_{}".format(upper)]).all()

    reduced = copy.deepcopy(small_config)
    reduced["nonconsent"]["end_percent"] = 80
    pd.testing.assert_frame_equal(
        indicators.drop(columns="gdpr_100"),
        generate_nonconsent(reduced),
    )


def test_independent_scenarios_are_exact_and_stable(small_config):
    small_config["nonconsent"]["assignment"] = "independent"
    indicators = generate_nonconsent(small_config)
    assert int(indicators["gdpr_40"].sum()) == 80
    assert int(indicators["gdpr_60"].sum()) == 120
    reduced = copy.deepcopy(small_config)
    reduced["nonconsent"]["end_percent"] = 80
    pd.testing.assert_frame_equal(
        indicators.drop(columns="gdpr_100"),
        generate_nonconsent(reduced),
    )


def test_split_partitions_ids_without_stratification(small_config):
    result = split_data(
        generate_features(small_config), generate_nonconsent(small_config), small_config
    )
    train_ids = result["train_features"]["id_row"].tolist()
    test_ids = result["test_features"]["id_row"].tolist()
    assert len(train_ids) == 150
    assert len(test_ids) == 50
    assert not set(train_ids) & set(test_ids)
    assert set(train_ids) | set(test_ids) == set(range(1, 201))
    assert train_ids == sorted(train_ids)
    assert test_ids == sorted(test_ids)
    assert result["summary"]["scenario_rates"]["gdpr_20"]["overall"] == 0.2


def test_preprocessor_uses_training_only_and_rejects_bad_schema(small_config):
    train = pd.DataFrame({"ft_1": [1.0, 3.0, pd.NA], "ft_2": [2.0, 2.0, 2.0], "ft_3": [0.0, 1.0, 2.0]})
    fitted = fit_preprocessor(train, small_config)
    assert fitted.imputer.statistics_[0] == 2.0
    assert fitted.scaler.mean_[0] == 2.0
    transformed = fitted.transform(pd.DataFrame({"ft_1": [1000.0], "ft_2": [2.0], "ft_3": [1.0]}))
    assert transformed[0, 0] > 100
    with pytest.raises(ValueError, match="fitted order"):
        fitted.transform(train[["ft_2", "ft_1", "ft_3"]])
    with pytest.raises(ValueError, match="entirely missing"):
        fit_preprocessor(
            pd.DataFrame({"ft_1": [np.nan, np.nan], "ft_2": [1.0, 2.0], "ft_3": [1, 2]}),
            small_config,
        )
