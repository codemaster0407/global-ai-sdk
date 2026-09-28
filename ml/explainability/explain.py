"""Model explainability for pipelines built by ``ml.classical.training``.

- ``model_feature_importance``: the model's own importances / coefficients (fast, global)
- ``permutation_importance_df``: model-agnostic, measured on held-out data
- ``shap_explain``: per-prediction attributions, plus global SHAP importance and plots
"""
import os
from typing import List, Optional

import matplotlib

matplotlib.use("Agg")  # save plots without a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline


def _split_pipeline(pipeline: Pipeline):
    """Return (preprocessor, model) from a preprocess + model pipeline."""
    return pipeline.named_steps["preprocess"], pipeline.named_steps["model"]


def transformed_features(pipeline: Pipeline, X: pd.DataFrame) -> pd.DataFrame:
    """``X`` as the model sees it (after imputation / scaling / one-hot)."""
    preprocess, _ = _split_pipeline(pipeline)
    Xt = preprocess.transform(X)
    if not isinstance(Xt, pd.DataFrame):
        Xt = pd.DataFrame(Xt, columns=preprocess.get_feature_names_out(), index=X.index)
    return Xt


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------
def model_feature_importance(pipeline: Pipeline, top_k: Optional[int] = None) -> pd.DataFrame:
    """Native importances: tree ``feature_importances_`` or |linear coefficients|.

    Tree importances are biased towards high-cardinality features; prefer
    permutation or SHAP importance when that matters.
    """
    preprocess, model = _split_pipeline(pipeline)
    names = preprocess.get_feature_names_out()

    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    elif hasattr(model, "coef_"):
        coef = np.atleast_2d(model.coef_)
        values = np.abs(coef).mean(axis=0)  # mean over classes for multiclass
    else:
        raise ValueError(f"{type(model).__name__} has no native feature importance")

    df = pd.DataFrame({"feature": names, "importance": values})
    df["importance"] = df["importance"] / df["importance"].sum()
    df = df.sort_values("importance", ascending=False).reset_index(drop=True)
    return df.head(top_k) if top_k else df


def permutation_importance_df(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    scoring: Optional[str] = None,
    n_repeats: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """Drop in score when each *raw* input column is shuffled - run on validation data."""
    result = permutation_importance(
        pipeline, X, y, scoring=scoring, n_repeats=n_repeats,
        random_state=random_state, n_jobs=-1,
    )
    return pd.DataFrame({
        "feature": X.columns,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------------
def shap_explain(
    pipeline: Pipeline,
    X: pd.DataFrame,
    background: Optional[pd.DataFrame] = None,
    max_samples: int = 500,
    class_index: int = 1,
    random_state: int = 42,
) -> shap.Explanation:
    """SHAP values for ``X`` (in transformed feature space).

    Tree models use the exact, fast TreeExplainer; linear models use
    LinearExplainer; anything else falls back to the model-agnostic
    (slower) permutation explainer.  For classifiers that return one set of
    values per class, ``class_index`` selects which class to explain.
    """
    _, model = _split_pipeline(pipeline)
    if len(X) > max_samples:
        X = X.sample(max_samples, random_state=random_state)
    Xt = transformed_features(pipeline, X)

    bg = transformed_features(pipeline, background) if background is not None else Xt
    bg = shap.sample(bg, min(100, len(bg)), random_state=random_state)

    model_name = type(model).__name__.lower()
    if "xgb" in model_name:
        explanation = _xgboost_shap(model, Xt)
        if explanation.values.ndim == 3:
            explanation = explanation[:, :, class_index]
        return explanation
    if any(t in model_name for t in ("lgbm", "forest")):
        explainer = shap.TreeExplainer(model)
    elif hasattr(model, "coef_"):
        explainer = shap.LinearExplainer(model, bg)
    else:
        predict = model.predict_proba if hasattr(model, "predict_proba") else model.predict
        explainer = shap.Explainer(predict, bg)

    explanation = explainer(Xt)

    # (n_samples, n_features, n_classes) -> pick one class
    if explanation.values.ndim == 3:
        explanation = explanation[:, :, class_index]
    return explanation


def _xgboost_shap(model, Xt: pd.DataFrame) -> shap.Explanation:
    """Exact TreeSHAP computed by XGBoost itself.

    Avoids shap.TreeExplainer, which fails to parse XGBoost >= 3 models
    (``base_score`` is stored as '[5E-1]').  Output is in log-odds / margin space.
    """
    import xgboost as xgb

    contribs = model.get_booster().predict(xgb.DMatrix(Xt), pred_contribs=True)
    if contribs.ndim == 3:  # multiclass: (n, n_classes, n_features + 1)
        contribs = contribs.transpose(0, 2, 1)  # -> (n, n_features + 1, n_classes)
    return shap.Explanation(
        values=contribs[:, :-1],
        base_values=contribs[:, -1],  # last column is the bias term
        data=Xt.values,
        feature_names=list(Xt.columns),
    )


def shap_importance(explanation: shap.Explanation) -> pd.DataFrame:
    """Global importance: mean |SHAP value| per feature."""
    return pd.DataFrame({
        "feature": explanation.feature_names,
        "mean_abs_shap": np.abs(explanation.values).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def explain_row(explanation: shap.Explanation, row: int = 0, top_k: int = 5) -> List[dict]:
    """Top features pushing one prediction up or down."""
    values = explanation.values[row]
    order = np.argsort(-np.abs(values))[:top_k]
    return [
        {
            "feature": explanation.feature_names[i],
            "value": float(explanation.data[row][i]),
            "shap": round(float(values[i]), 4),
            "direction": "increases" if values[i] > 0 else "decreases",
        }
        for i in order
    ]


def save_shap_plots(explanation: shap.Explanation, out_dir: str = "ml/artifacts/shap", max_display: int = 15) -> List[str]:
    """Save beeswarm (global), bar (global) and waterfall (first row) plots."""
    os.makedirs(out_dir, exist_ok=True)
    plots = {
        "beeswarm": lambda: shap.plots.beeswarm(explanation, max_display=max_display, show=False),
        "bar": lambda: shap.plots.bar(explanation, max_display=max_display, show=False),
        "waterfall_row0": lambda: shap.plots.waterfall(explanation[0], max_display=max_display, show=False),
    }

    paths = []
    for name, draw in plots.items():
        plt.figure()
        draw()
        path = os.path.join(out_dir, f"{name}.png")
        plt.savefig(path, bbox_inches="tight", dpi=120)
        plt.close("all")
        paths.append(path)
    return paths


if __name__ == "__main__":
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split

    from ml.classical.training import train_model

    data = load_breast_cancer(as_frame=True).frame
    result = train_model(data, target="target", model_name="lightgbm")
    pipeline = result.model

    X = data.drop(columns="target")
    _, X_val, _, y_val = train_test_split(X, data["target"], test_size=0.2, random_state=42, stratify=data["target"])

    print("Model importance:\n", model_feature_importance(pipeline, top_k=5), "\n")
    print("Permutation importance:\n", permutation_importance_df(pipeline, X_val, y_val, scoring="roc_auc").head(5), "\n")

    explanation = shap_explain(pipeline, X_val)
    print("SHAP importance:\n", shap_importance(explanation).head(5), "\n")
    print("Why row 0:", explain_row(explanation, 0, top_k=3))
    print("Plots:", save_shap_plots(explanation))
