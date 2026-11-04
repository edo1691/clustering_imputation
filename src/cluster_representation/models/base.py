"""Base API shared by fitted clustering models."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict

import joblib


class ClusterModel(ABC):
    labels_: Any

    @abstractmethod
    def fit(self, X: Any) -> "ClusterModel":
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def metadata(self) -> Dict[str, Any]:
        raise NotImplementedError

    def save(self, path: Any) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, target)

    @classmethod
    def load(cls, path: Any) -> "ClusterModel":
        model = joblib.load(path)
        if not isinstance(model, cls):
            raise TypeError("Serialized model is not a {}".format(cls.__name__))
        return model
