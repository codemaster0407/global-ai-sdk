"""Train scikit-learn / XGBoost / LightGBM models behind one interface."""
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from ml.features.pipelines import build_preprocessor, infer_column_types

TREE_MODELS = {"xgboost", "lightgbm", "random_forest"}


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------
def build_model(name: str, task: str = "classification", **params):
    """Return an unfitted estimator for ``name`` and ``task``.

    XGBoost and LightGBM are imported lazily so the scikit-learn models work
    without them installed.
    """
    is_clf = task == "classification"

    if name == "logistic_regression" and is_clf:
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(max_iter=1000, **params)
    if name == "linear_regression" and not is_clf:
        from sklearn.linear_model import Ridge
        return Ridge(**params)
    if name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        cls = RandomForestClassifier if is_clf else RandomForestRegressor
        return cls(n_estimators=300, n_jobs=-1, **params)
    if name == "xgboost":
        import xgboost as xgb
        cls = xgb.XGBClassifier if is_clf else xgb.XGBRegressor
        return cls(n_estimators=500, learning_rate=0.05, max_depth=6, **params)
    if name == "lightgbm":
        import lightgbm as lgb
        cls = lgb.LGBMClassifier if is_clf else lgb.LGBMRegressor
        return cls(n_estimators=500, learning_rate=0.05, verbose=-1, **params)

    raise ValueError(f"Unsupported model '{name}' for task '{task}'")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(model: Pipeline, X: pd.DataFrame, y: pd.Series, task: str) -> Dict[str, float]:
    """Compute standard metrics for a fitted pipeline on a held-out set."""
    preds = model.predict(X)

    if task == "classification":
        metrics = {
            "accuracy": accuracy_score(y, preds),
            "f1_weighted": f1_score(y, preds, average="weighted"),
        }
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
            if proba.shape[1] == 2:
                metrics["roc_auc"] = roc_auc_score(y, proba[:, 1])
            else:
                metrics["roc_auc_ovr"] = roc_auc_score(y, proba, multi_class="ovr")
    else:
        metrics = {
            "mae": mean_absolute_error(y, preds),
            "rmse": float(np.sqrt(mean_squared_error(y, preds))),
            "r2": r2_score(y, preds),
        }
    return {k: round(float(v), 4) for k, v in metrics.items()}


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------
@dataclass
class TrainResult:
    model: Pipeline
    metrics: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)


def train_model(
    df: pd.DataFrame,
    target: str,
    model_name: str = "lightgbm",
    task: str = "classification",
    numeric_cols: Optional[List[str]] = None,
    categorical_cols: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    params: Optional[Dict[str, Any]] = None,
    output_path: Optional[str] = None,
) -> TrainResult:
    """Train a preprocessing + model pipeline and optionally save it.

    Parameters
    ----------
    df: pd.DataFrame
        Training data including the ``target`` column.
    model_name: str
        One of ``logistic_regression``, ``linear_regression``,
        ``random_forest``, ``xgboost``, ``lightgbm``.
    task: str
        ``classification`` or ``regression``.
    numeric_cols / categorical_cols: list | None
        Feature columns.  Inferred from dtypes when both are ``None``.
    exclude: list | None
        Columns to ignore during inference of feature columns (ids, dates).
    output_path: str | None
        When set, the fitted pipeline and its metadata are saved with joblib.
    """
    if numeric_cols is None and categorical_cols is None:
        numeric_cols, categorical_cols = infer_column_types(df, target, exclude)
    numeric_cols = numeric_cols or []
    categorical_cols = categorical_cols or []
    feature_cols = numeric_cols + categorical_cols

    X, y = df[feature_cols], df[target]

    # XGBoost needs integer class labels, so encode string targets up front
    classes = None
    if task == "classification" and not pd.api.types.is_numeric_dtype(y):
        classes = sorted(y.unique().tolist())
        y = y.map({c: i for i, c in enumerate(classes)})

    stratify = y if task == "classification" else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    pipeline = Pipeline([
        ("preprocess", build_preprocessor(
            numeric_cols, categorical_cols, scale=model_name not in TREE_MODELS
        )),
        ("model", build_model(model_name, task, **(params or {}))),
    ])
    pipeline.fit(X_train, y_train)

    metrics = evaluate(pipeline, X_test, y_test, task)
    metadata = {
        "model_name": model_name,
        "task": task,
        "target": target,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "classes": classes,
        "metrics": metrics,
        "n_train": len(X_train),
        "n_test": len(X_test),
    }
    print(f"[{model_name}] {task} metrics: {json.dumps(metrics)}")

    if output_path:
        save_model(pipeline, metadata, output_path)

    return TrainResult(model=pipeline, metrics=metrics, metadata=metadata)


def save_model(pipeline: Pipeline, metadata: Dict[str, Any], output_path: str) -> None:
    """Save the pipeline and metadata together as a single joblib bundle."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    joblib.dump({"pipeline": pipeline, "metadata": metadata}, output_path)
    print(f"Model saved to {output_path}")


if __name__ == "__main__":
    from sklearn.datasets import load_breast_cancer

    data = load_breast_cancer(as_frame=True).frame
    train_model(
        data,
        target="target",
        model_name="lightgbm",
        task="classification",
        output_path="ml/artifacts/breast_cancer_lgbm.joblib",
    )
