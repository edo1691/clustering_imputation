"""Deterministic train/test splitting with ID-safe scenario joins."""

from typing import Any, Dict, Mapping

import pandas as pd
from sklearn.model_selection import train_test_split


def split_data(
    features: pd.DataFrame, indicators: pd.DataFrame, config: Mapping[str, Any]
) -> Dict[str, Any]:
    id_column = config["dataset"]["id_column"]
    feature_columns = [feature["name"] for feature in config["features"]]
    expected_feature_columns = [id_column] + feature_columns
    expected_indicator_columns = [id_column] + [
        "{}_{}".format(config["nonconsent"]["prefix"], percent)
        for percent in range(
            config["nonconsent"]["start_percent"],
            config["nonconsent"]["end_percent"] + 1,
            config["nonconsent"]["step_percent"],
        )
    ]
    _validate_table(features, expected_feature_columns, id_column, "features")
    _validate_table(indicators, expected_indicator_columns, id_column, "indicators")
    if features[id_column].duplicated().any() or indicators[id_column].duplicated().any():
        raise ValueError("Feature and indicator IDs must be unique")
    if set(features[id_column]) != set(indicators[id_column]):
        raise ValueError("Feature and indicator IDs must match exactly")

    all_ids = features[id_column].tolist()
    train_ids, test_ids = train_test_split(
        all_ids, test_size=config["split"]["test_size"], random_state=config["split"]["seed"]
    )
    train_set, test_set = set(train_ids), set(test_ids)
    if train_set & test_set or train_set | test_set != set(all_ids):
        raise RuntimeError("Train/test split did not partition all IDs")
    train_ids = [identifier for identifier in all_ids if identifier in train_set]
    test_ids = [identifier for identifier in all_ids if identifier in test_set]

    def select(frame: pd.DataFrame, ids: list) -> pd.DataFrame:
        return frame.set_index(id_column).loc[ids].reset_index()

    train_features = select(features, train_ids)
    test_features = select(features, test_ids)
    train_indicators = select(indicators, train_ids)
    test_indicators = select(indicators, test_ids)
    rates = {}
    for column in expected_indicator_columns[1:]:
        rates[column] = {
            "overall": float(indicators[column].mean()),
            "train": float(train_indicators[column].mean()),
            "test": float(test_indicators[column].mean()),
        }
    return {
        "train_features": train_features,
        "test_features": test_features,
        "train_indicators": train_indicators,
        "test_indicators": test_indicators,
        "summary": {
            "n_train": len(train_ids),
            "n_test": len(test_ids),
            "scenario_rates": rates,
        },
    }


def _validate_table(
    frame: pd.DataFrame, expected_columns: list, id_column: str, table_name: str
) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("{} must be a pandas DataFrame".format(table_name))
    if frame.columns.duplicated().any():
        raise ValueError("{} contains duplicate columns".format(table_name))
    if list(frame.columns) != expected_columns:
        raise ValueError(
            "{} columns must be exactly {}".format(table_name, expected_columns)
        )
    if frame[id_column].isna().any():
        raise ValueError("{} contains missing IDs".format(table_name))
