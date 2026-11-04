import numpy as np
import pandas as pd
import pytest

from cluster_representation.models import create_model
from cluster_representation.models.constrained_kmeans import ConstrainedKMeans
from cluster_representation.prototypes import build_prototypes
from cluster_representation.transformation import transform_test


def test_constrained_kmeans_enforces_minimum_and_roundtrips(small_config, tmp_path):
    model = create_model(small_config.model_config)
    rng = np.random.default_rng(42)
    X = rng.normal(size=(90, 3))
    model.fit(X)
    counts = np.bincount(model.labels_)[1:]
    assert len(counts) == 3
    assert counts.min() >= 5
    assert set(model.predict(X)).issubset({1, 2, 3})

    path = tmp_path / "model.joblib"
    model.save(path)
    loaded = ConstrainedKMeans.load(path)
    np.testing.assert_array_equal(model.predict(X), loaded.predict(X))
    np.testing.assert_array_equal(model.labels_, loaded.labels_)


def test_constrained_kmeans_rejects_infeasible_minimum(small_config):
    config = dict(small_config.model_config)
    config["parameters"] = dict(config["parameters"], n_clusters=3)
    config["constraints"] = dict(config["constraints"], min_cluster_size=31)
    model = create_model(config)
    with pytest.raises(ValueError, match="Infeasible"):
        model.fit(np.ones((90, 2)))


def test_future_coclustering_is_registered_but_explicitly_unimplemented():
    model = create_model(
        {
            "name": "robust_constrained_coclustering",
            "parameters": {
                "n_row_clusters": 2,
                "n_column_clusters": 2,
                "random_state": 1,
                "max_iter": 10,
                "n_init": 1,
            },
            "constraints": {"min_row_cluster_size": 2},
            "trimming": {"row_fraction": 0.1, "column_fraction": 0.1},
        }
    )
    with pytest.raises(NotImplementedError, match="template only"):
        model.fit(np.ones((10, 4)))
    with pytest.raises(NotImplementedError, match="prediction is not implemented"):
        model.predict(np.ones((2, 4)))


def test_constrained_coclustering_is_registered_and_unimplemented():
    model_config = {
        "name": "constrained_coclustering",
        "parameters": {
            "n_row_clusters": 3,
            "n_column_clusters": 2,
            "random_state": 12345,
            "max_iter": 100,
            "n_init": 5,
        },
        "constraints": {"min_row_cluster_size": 10},
    }
    model = create_model(model_config)
    assert model.metadata()["implemented"] is False
    assert not hasattr(model, "row_trimming_mask_")
    assert not hasattr(model, "column_trimming_mask_")
    with pytest.raises(NotImplementedError, match="configuration template only"):
        model.fit(np.ones((20, 4)))
    with pytest.raises(NotImplementedError, match="prediction is not implemented"):
        model.predict(np.ones((2, 4)))


def test_prototypes_use_fitted_memberships_and_original_unit_medians(small_config):
    train = pd.DataFrame(
        {
            "id_row": [1, 2, 3, 4],
            "ft_1": [1.0, 9.0, 3.0, 7.0],
            "ft_2": [100.0, 200.0, 300.0, 400.0],
            "ft_3": [1, 2, 3, 4],
        }
    )
    result = build_prototypes(train, [1, 1, 2, 2], small_config, 2)
    assert result["assignments"]["cluster"].tolist() == [1, 1, 2, 2]
    assert result["prototypes"].loc[0, "ft_1"] == 5.0
    assert result["prototypes"].loc[0, "ft_2"] == 150.0
    assert result["prototypes"].loc[0, "ft_3"] == 1.5
    assert result["prototypes"]["n_clients"].tolist() == [2, 2]


def test_transform_replaces_all_features_and_preserves_others(small_config):
    test = pd.DataFrame(
        {
            "id_row": [10, 11],
            "ft_1": [5.0, 8.0],
            "ft_2": [10.0, 20.0],
            "ft_3": [2, 4],
        }
    )
    indicators = pd.DataFrame({"id_row": [10, 11], "gdpr_20": [1, 0]})
    prototypes = pd.DataFrame(
        {
            "cluster": [1, 2],
            "n_clients": [10, 10],
            "ft_1": [1.5, 6.5],
            "ft_2": [15.0, 35.0],
            "ft_3": [1.5, 3.5],
        }
    )
    assignments = pd.DataFrame({"id_row": [10, 11], "cluster": [1, 2]})
    result = transform_test(
        test, indicators, prototypes, assignments, small_config
    )["transformed"]
    assert result.columns.tolist() == ["id_row", "gdpr", "cluster", "ft_1", "ft_2", "ft_3"]
    assert result.loc[0, ["ft_1", "ft_2", "ft_3"]].tolist() == [1.5, 15.0, 1.5]
    assert result.loc[1, ["ft_1", "ft_2", "ft_3"]].tolist() == [8.0, 20.0, 4.0]
    assert result["id_row"].tolist() == [10, 11]


def test_transform_rejects_duplicate_or_missing_join_ids(small_config):
    test = pd.DataFrame(
        {"id_row": [1, 2], "ft_1": [1, 2], "ft_2": [2, 3], "ft_3": [1, 2]}
    )
    indicators = pd.DataFrame({"id_row": [1, 1], "gdpr_20": [0, 1]})
    prototypes = pd.DataFrame(
        {"cluster": [1], "n_clients": [2], "ft_1": [1], "ft_2": [2], "ft_3": [1]}
    )
    assignments = pd.DataFrame({"id_row": [1, 2], "cluster": [1, 1]})
    with pytest.raises(ValueError, match="duplicate IDs"):
        transform_test(test, indicators, prototypes, assignments, small_config)
