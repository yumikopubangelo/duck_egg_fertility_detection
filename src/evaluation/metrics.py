"""Classification evaluation metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def _to_numpy(values: Iterable[int] | np.ndarray) -> np.ndarray:
    arr = np.asarray(list(values) if not isinstance(values, np.ndarray) else values)
    return arr.astype(np.int64).ravel()


@dataclass
class Accuracy:
    """Accuracy metric."""

    @staticmethod
    def compute(y_true: Iterable[int] | np.ndarray, y_pred: Iterable[int] | np.ndarray) -> float:
        return float(accuracy_score(_to_numpy(y_true), _to_numpy(y_pred)))


@dataclass
class Precision:
    """Precision metric."""

    @staticmethod
    def compute(
        y_true: Iterable[int] | np.ndarray,
        y_pred: Iterable[int] | np.ndarray,
        average: str = "binary",
        zero_division: int = 0,
    ) -> float:
        return float(
            precision_score(
                _to_numpy(y_true),
                _to_numpy(y_pred),
                average=average,
                zero_division=zero_division,
            )
        )


@dataclass
class Recall:
    """Recall metric."""

    @staticmethod
    def compute(
        y_true: Iterable[int] | np.ndarray,
        y_pred: Iterable[int] | np.ndarray,
        average: str = "binary",
        zero_division: int = 0,
    ) -> float:
        return float(
            recall_score(
                _to_numpy(y_true),
                _to_numpy(y_pred),
                average=average,
                zero_division=zero_division,
            )
        )


@dataclass
class F1Score:
    """F1 score metric."""

    @staticmethod
    def compute(
        y_true: Iterable[int] | np.ndarray,
        y_pred: Iterable[int] | np.ndarray,
        average: str = "binary",
        zero_division: int = 0,
    ) -> float:
        return float(
            f1_score(
                _to_numpy(y_true),
                _to_numpy(y_pred),
                average=average,
                zero_division=zero_division,
            )
        )


@dataclass
class ConfusionMatrix:
    """Confusion matrix metric."""

    @staticmethod
    def compute(
        y_true: Iterable[int] | np.ndarray,
        y_pred: Iterable[int] | np.ndarray,
        labels: List[int] | None = None,
    ) -> np.ndarray:
        return confusion_matrix(_to_numpy(y_true), _to_numpy(y_pred), labels=labels)


@dataclass
class ClassificationReport:
    """Classification report utility."""

    @staticmethod
    def compute(
        y_true: Iterable[int] | np.ndarray,
        y_pred: Iterable[int] | np.ndarray,
        target_names: List[str] | None = None,
        output_dict: bool = True,
    ) -> Dict:
        return classification_report(
            _to_numpy(y_true),
            _to_numpy(y_pred),
            target_names=target_names,
            output_dict=output_dict,
            zero_division=0,
        )
