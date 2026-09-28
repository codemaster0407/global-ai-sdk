"""Class imbalance handling: class weights, resampling (SMOTE etc.) and threshold tuning.

Rule of thumb: try class weights first (cheap, no synthetic data), then
resampling, and always tune the decision threshold on a validation set -
the default 0.5 is rarely right for an imbalanced problem.
"""
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    roc_curve,
)
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight

from ml.classical.training import TREE_MODELS, build_model
from ml.features.pipelines import build_preprocessor


# ---------------------------------------------------------------------------
# Inspection
# ---------------------------------------------------------------------------
def class_distribution(y: pd.Series) -> pd.DataFrame:
    counts = y.value_counts()
    return pd.DataFrame({"count": counts, "ratio": (counts / counts.sum()).round(4)})


# ---------------------------------------------------------------------------
# Class weights
# ---------------------------------------------------------------------------
def compute_class_weights(y: pd.Series) -> Dict[Any, float]:
    """``n_samples / (n_classes * class_count)`` for each class."""
    classes = np.unique(y)
    weights = compute_class_weight("balanced", classes=classes, y=y)
    return dict(zip(classes.tolist(), weights.round(4).tolist()))


def class_weight_params(model_name: str, y: pd.Series) -> Dict[str, Any]:
    """Model params that reweight classes; pass as ``params=`` to ``train_model``.

    XGBoost has no ``class_weight`` - binary uses ``scale_pos_weight``;
    for multiclass XGBoost, pass ``sample_weights(y)`` to ``fit`` instead.
    """
    if model_name == "xgboost":
        if y.nunique() != 2:
            raise ValueError("Multiclass XGBoost: use sample_weights(y) in fit() instead")
        counts = y.value_counts()
        return {"scale_pos_weight": float(counts.max() / counts.min())}
    if model_name in {"lightgbm", "random_forest", "logistic_regression"}:
        return {"class_weight": "balanced"}
    raise ValueError(f"Class weights not supported for '{model_name}'")


def sample_weights(y: pd.Series) -> np.ndarray:
    """Per-row weights, for ``pipeline.fit(X, y, model__sample_weight=...)``."""
    return compute_sample_weight("balanced", y)


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------
SAMPLERS = {
    "smote": lambda rs: SMOTE(random_state=rs),
    "adasyn": lambda rs: ADASYN(random_state=rs),
    "random_over": lambda rs: RandomOverSampler(random_state=rs),
    "random_under": lambda rs: RandomUnderSampler(random_state=rs),
    "smote_tomek": lambda rs: SMOTETomek(random_state=rs),
}


def build_resampled_pipeline(
    model_name: str,
    numeric_cols: List[str],
    categorical_cols: List[str],
    sampler: str = "smote",
    params: Optional[Dict[str, Any]] = None,
    random_state: int = 42,
) -> ImbPipeline:
    """Preprocess -> resample -> model.

    The imblearn pipeline only resamples during ``fit``, so validation / test
    data and ``predict`` calls are never touched - resampling before a split
    would leak synthetic copies of test rows into training.
    """
    return ImbPipeline([
        ("preprocess", build_preprocessor(
            numeric_cols, categorical_cols, scale=model_name not in TREE_MODELS
        )),
        ("sampler", SAMPLERS[sampler](random_state)),
        ("model", build_model(model_name, "classification", **(params or {}))),
    ])


# ---------------------------------------------------------------------------
# Threshold tuning (binary)
# ---------------------------------------------------------------------------
def tune_threshold(
    y_true: np.ndarray,
    proba: np.ndarray,
    metric: str = "f1",
    beta: float = 2.0,
    min_recall: float = 0.8,
    min_precision: float = 0.8,
) -> Dict[str, float]:
    """Pick the probability cut-off for the positive class on a validation set.

    metric:
        ``f1``                  - best F1
        ``fbeta``               - best F-beta (beta > 1 favours recall)
        ``precision_at_recall`` - highest precision with recall >= ``min_recall``
        ``recall_at_precision`` - highest recall with precision >= ``min_precision``
        ``youden``              - maximises TPR - FPR on the ROC curve
    """
    if metric == "youden":
        fpr, tpr, thresholds = roc_curve(y_true, proba)
        best = int(np.argmax(tpr - fpr))
        return {"threshold": float(thresholds[best]), "score": float(tpr[best] - fpr[best])}

    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    # The last precision/recall pair has no threshold
    precision, recall = precision[:-1], recall[:-1]

    if metric == "f1":
        scores = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    elif metric == "fbeta":
        b2 = beta ** 2
        scores = (1 + b2) * precision * recall / np.clip(b2 * precision + recall, 1e-12, None)
    elif metric == "precision_at_recall":
        scores = np.where(recall >= min_recall, precision, -1)
    elif metric == "recall_at_precision":
        scores = np.where(precision >= min_precision, recall, -1)
    else:
        raise ValueError(f"Unknown metric '{metric}'")

    best = int(np.argmax(scores))
    if scores[best] < 0:
        raise ValueError(f"No threshold satisfies the {metric} constraint")

    return {
        "threshold": float(thresholds[best]),
        "score": float(scores[best]),
        "precision": float(precision[best]),
        "recall": float(recall[best]),
    }


def apply_threshold(proba: np.ndarray, threshold: float) -> np.ndarray:
    return (proba >= threshold).astype(int)


def imbalance_report(y_true: np.ndarray, proba: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """Metrics that stay meaningful under imbalance (accuracy does not)."""
    preds = apply_threshold(proba, threshold)
    return {
        "threshold": round(threshold, 4),
        "pr_auc": round(average_precision_score(y_true, proba), 4),
        "f1": round(f1_score(y_true, preds), 4),
        "f2": round(fbeta_score(y_true, preds, beta=2), 4),
        "recall": round(float(preds[y_true == 1].mean()), 4),
        "precision": round(float(y_true[preds == 1].mean()) if preds.sum() else 0.0, 4),
    }


if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    from ml.classical.training import build_pipeline

    X, y = make_classification(n_samples=5000, n_features=20, n_informative=6,
                               weights=[0.95, 0.05], random_state=0)
    X = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    y = pd.Series(y)
    cols = X.columns.tolist()
    print(class_distribution(y), "\n")

    # train / validation (threshold tuning) / test (final report)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, stratify=y_train, random_state=0)

    candidates = {
        "baseline": build_pipeline("lightgbm", "classification", cols, []),
        "class_weights": build_pipeline("lightgbm", "classification", cols, [],
                                        params=class_weight_params("lightgbm", y_train)),
        "smote": build_resampled_pipeline("lightgbm", cols, [], sampler="smote"),
    }
    for name, pipeline in candidates.items():
        pipeline.fit(X_train, y_train)
        val_proba = pipeline.predict_proba(X_val)[:, 1]
        test_proba = pipeline.predict_proba(X_test)[:, 1]

        tuned = tune_threshold(y_val.values, val_proba, metric="f1")
        print(f"{name:<14} @0.5   ", imbalance_report(y_test.values, test_proba))
        print(f"{name:<14} tuned  ", imbalance_report(y_test.values, test_proba, tuned["threshold"]))
