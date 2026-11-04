"""Reusable validation for saved workflow artifacts."""

from typing import Iterable

import pandas as pd

from .config import LoadedConfig
from .storage import require_stage


def validate_stage_artifacts(
    config: LoadedConfig,
    stage: str,
    fingerprint: str,
    artifact_paths: Iterable[str],
) -> None:
    """Reject missing or configuration-incompatible prerequisite artifacts."""
    require_stage(config, stage, fingerprint, artifact_paths)


def validate_frame_schema(
    frame: pd.DataFrame, expected_columns: Iterable[str], artifact_name: str
) -> None:
    expected = list(expected_columns)
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("{} must be a pandas DataFrame".format(artifact_name))
    if frame.columns.duplicated().any():
        raise ValueError("{} has duplicate column names".format(artifact_name))
    if list(frame.columns) != expected:
        raise ValueError(
            "{} schema mismatch: expected {}, got {}".format(
                artifact_name, expected, list(frame.columns)
            )
        )


def validate_id_alignment(
    left: pd.DataFrame, right: pd.DataFrame, id_column: str, description: str
) -> None:
    for frame in (left, right):
        if id_column not in frame or frame[id_column].isna().any():
            raise ValueError("{} has missing or null IDs".format(description))
        if frame[id_column].duplicated().any():
            raise ValueError("{} has duplicate IDs".format(description))
    if left[id_column].tolist() != right[id_column].tolist():
        raise ValueError("{} ID order does not match".format(description))
