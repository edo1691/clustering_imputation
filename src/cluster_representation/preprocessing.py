"""Training-only imputation and scaling for model inputs."""

from typing import Any, List, Mapping, Optional

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


class FeaturePreprocessor:
    """Fitted numeric transformation that enforces the original feature schema."""

    def __init__(self, feature_names: List[str], scaling: str) -> None:
        self.feature_names = list(feature_names)
        self.scaling = scaling
        self.imputer: Optional[SimpleImputer] = None
        self.scaler: Optional[StandardScaler] = None

    def fit(self, frame: pd.DataFrame) -> "FeaturePreprocessor":
        _validate_features(frame, self.feature_names)
        values = _numeric_values(frame, self.feature_names)
        if np.isnan(values).all(axis=0).any():
            raise ValueError("Cannot median-impute a feature that is entirely missing")
        self.imputer = SimpleImputer(strategy="median")
        imputed = self.imputer.fit_transform(values)
        if self.scaling == "standard":
            self.scaler = StandardScaler()
            self.scaler.fit(imputed)
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if self.imputer is None:
            raise RuntimeError("Preprocessor has not been fitted")
        _validate_features(frame, self.feature_names)
        values = self.imputer.transform(_numeric_values(frame, self.feature_names))
        if self.scaler is not None:
            values = self.scaler.transform(values)
        return np.asarray(values, dtype=float)


def fit_preprocessor(
    train_features: pd.DataFrame, config: Mapping[str, Any]
) -> FeaturePreprocessor:
    names = [feature["name"] for feature in config["features"]]
    preprocessor = FeaturePreprocessor(names, config["preprocessing"]["scaling"])
    return preprocessor.fit(train_features)


def _validate_features(frame: pd.DataFrame, feature_names: List[str]) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Features must be provided as a pandas DataFrame")
    if frame.columns.duplicated().any():
        raise ValueError("Feature input contains duplicate columns")
    if list(frame.columns) != feature_names:
        raise ValueError(
            "Feature columns must match the fitted order {}; got {}".format(
                feature_names, list(frame.columns)
            )
        )


def _numeric_values(frame: pd.DataFrame, feature_names: List[str]) -> np.ndarray:
    try:
        values = frame.loc[:, feature_names].astype("Float64").to_numpy(
            dtype=float, na_value=np.nan
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("All model features must be numeric") from exc
    if np.isinf(values).any():
        raise ValueError("Features must not contain infinite values")
    return values
