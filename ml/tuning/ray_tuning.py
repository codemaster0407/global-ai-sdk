"""Distributed hyperparameter tuning / AutoML with Ray Tune.

Use this over ``optuna_tuning`` when trials should run in parallel across
cores or a Ray cluster.  Also tries several model families (AutoML style).

One-time setup:
    pip install "ray[tune]" optuna
"""
from functools import partial
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
from sklearn.base import clone
from sklearn.metrics import get_scorer
from sklearn.model_selection import KFold, StratifiedKFold

from ml.classical.training import build_pipeline, prepare_data
from ml.tuning.optuna_tuning import default_scoring

# ---------------------------------------------------------------------------
# Search spaces (Ray Tune syntax)
# ---------------------------------------------------------------------------
SEARCH_SPACES: Dict[str, Dict[str, Any]] = {
    "lightgbm": {
        "n_estimators": tune.qrandint(100, 1000, 50),
        "learning_rate": tune.loguniform(1e-3, 0.3),
        "num_leaves": tune.randint(15, 256),
        "max_depth": tune.randint(3, 13),
        "min_child_samples": tune.randint(5, 101),
        "subsample": tune.uniform(0.5, 1.0),
        "colsample_bytree": tune.uniform(0.5, 1.0),
        "reg_lambda": tune.loguniform(1e-8, 10.0),
    },
    "xgboost": {
        "n_estimators": tune.qrandint(100, 1000, 50),
        "learning_rate": tune.loguniform(1e-3, 0.3),
        "max_depth": tune.randint(3, 11),
        "min_child_weight": tune.uniform(1.0, 10.0),
        "subsample": tune.uniform(0.5, 1.0),
        "colsample_bytree": tune.uniform(0.5, 1.0),
        "reg_lambda": tune.loguniform(1e-8, 10.0),
    },
    "random_forest": {
        "n_estimators": tune.qrandint(100, 600, 50),
        "max_depth": tune.randint(3, 31),
        "min_samples_leaf": tune.randint(1, 11),
        "max_features": tune.choice(["sqrt", "log2"]),
    },
    "logistic_regression": {"C": tune.loguniform(1e-4, 100.0)},
    "linear_regression": {"alpha": tune.loguniform(1e-4, 100.0)},
}


def _automl_space(trial, model_names: List[str]):
    """Conditional Optuna space: the model family is a hyperparameter, and each
    family only samples its own params.  Module-level so Ray can pickle it."""
    name = trial.suggest_categorical("model_name", model_names)
    if name == "lightgbm":
        trial.suggest_int("n_estimators", 100, 1000, step=50)
        trial.suggest_float("learning_rate", 1e-3, 0.3, log=True)
        trial.suggest_int("num_leaves", 15, 255)
    elif name == "xgboost":
        trial.suggest_int("n_estimators", 100, 1000, step=50)
        trial.suggest_float("learning_rate", 1e-3, 0.3, log=True)
        trial.suggest_int("max_depth", 3, 10)
    elif name == "random_forest":
        trial.suggest_int("n_estimators", 100, 600, step=50)
        trial.suggest_int("max_depth", 3, 30)
    elif name == "logistic_regression":
        trial.suggest_float("C", 1e-4, 100.0, log=True)
    elif name == "linear_regression":
        trial.suggest_float("alpha", 1e-4, 100.0, log=True)


def _cv_trainable(config, X, y, folds, task, numeric_cols, categorical_cols, scoring):
    """One Ray trial: cross-validate a sampled config, reporting after every fold."""
    config = dict(config)
    model_name = config.pop("model_name")
    if model_name == "lightgbm":
        config["subsample_freq"] = 1
    # Parallel Ray trials already use all cores; keep each model single-threaded
    if model_name in {"lightgbm", "xgboost", "random_forest"}:
        config["n_jobs"] = 1

    pipeline = build_pipeline(model_name, task, numeric_cols, categorical_cols, config)
    scorer = get_scorer(scoring)

    scores = []
    for train_idx, val_idx in folds:
        fold_model = clone(pipeline).fit(X.iloc[train_idx], y.iloc[train_idx])
        scores.append(scorer(fold_model, X.iloc[val_idx], y.iloc[val_idx]))
        # ASHA can stop weak trials early based on these intermediate results
        tune.report({"score": float(np.mean(scores))})


def tune_with_ray(
    df: pd.DataFrame,
    target: str,
    model_names: List[str] = ("lightgbm",),
    task: str = "classification",
    num_samples: int = 50,
    cv: int = 5,
    scoring: Optional[str] = None,
    max_concurrent: Optional[int] = None,
    cpus_per_trial: float = 1,
    storage_path: Optional[str] = None,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Search one or more model families at once.

    Passing several ``model_names`` turns this into a small AutoML: the model
    family is itself a hyperparameter, and Optuna picks which to explore.

    Returns ``{"model_name", "params", "score", "results"}`` for the best trial.
    """
    X, y, numeric_cols, categorical_cols, _ = prepare_data(df, target, task)
    scoring = scoring or default_scoring(task, y)

    splitter_cls = StratifiedKFold if task == "classification" else KFold
    folds = list(splitter_cls(n_splits=cv, shuffle=True, random_state=random_state).split(X, y))

    model_names = list(model_names)
    if len(model_names) == 1:
        param_space = {"model_name": model_names[0], **SEARCH_SPACES[model_names[0]]}
        search_alg = OptunaSearch(seed=random_state)
    else:
        param_space = None
        search_alg = OptunaSearch(
            space=partial(_automl_space, model_names=model_names),
            metric="score", mode="max", seed=random_state,
        )

    trainable = tune.with_resources(
        tune.with_parameters(
            _cv_trainable, X=X, y=y, folds=folds, task=task,
            numeric_cols=numeric_cols, categorical_cols=categorical_cols, scoring=scoring,
        ),
        {"cpu": cpus_per_trial},
    )

    tuner = tune.Tuner(
        trainable,
        param_space=param_space,
        tune_config=tune.TuneConfig(
            metric="score",
            mode="max",
            num_samples=num_samples,
            search_alg=search_alg,
            scheduler=ASHAScheduler(max_t=cv, grace_period=1, reduction_factor=2),
            max_concurrent_trials=max_concurrent,
        ),
        run_config=tune.RunConfig(storage_path=storage_path, verbose=1),
    )
    results = tuner.fit()

    best = results.get_best_result(metric="score", mode="max")
    params = dict(best.config)
    model_name = params.pop("model_name")
    print(f"Best model: {model_name}  {scoring}={best.metrics['score']:.4f}  params={params}")

    return {"model_name": model_name, "params": params, "score": best.metrics["score"], "results": results}


if __name__ == "__main__":
    from sklearn.datasets import load_breast_cancer

    data = load_breast_cancer(as_frame=True).frame
    tune_with_ray(
        data,
        target="target",
        model_names=["lightgbm", "xgboost", "logistic_regression"],
        num_samples=20,
        cv=3,
    )
