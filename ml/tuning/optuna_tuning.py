"""Hyperparameter tuning for the classical models with Optuna (TPE search + pruning)."""
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import optuna
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import get_scorer
from sklearn.model_selection import KFold, StratifiedKFold

from ml.classical.training import build_pipeline, prepare_data, train_model


# ---------------------------------------------------------------------------
# Search spaces
# ---------------------------------------------------------------------------
def suggest_params(trial: optuna.Trial, model_name: str) -> Dict[str, Any]:
    """Sample hyperparameters for ``model_name``; edit the ranges to fit your data."""
    if model_name == "lightgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 255),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "subsample_freq": 1,
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }
    if model_name == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }
    if model_name == "random_forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        }
    if model_name == "logistic_regression":
        return {"C": trial.suggest_float("C", 1e-4, 100.0, log=True)}
    if model_name == "linear_regression":
        return {"alpha": trial.suggest_float("alpha", 1e-4, 100.0, log=True)}
    if model_name == "tabpfn":
        # Pretrained, so only inference-time knobs: ensemble size and probability sharpness.
        # Gains are usually small; each trial re-runs TabPFN on every fold, so keep n_trials low.
        return {
            "n_estimators": trial.suggest_int("n_estimators", 1, 16),
            "softmax_temperature": trial.suggest_float("softmax_temperature", 0.5, 1.0),
            "average_before_softmax": trial.suggest_categorical("average_before_softmax", [False, True]),
        }

    raise ValueError(f"No search space defined for '{model_name}'")


def default_scoring(task: str, y: pd.Series) -> str:
    """Sklearn scorer name (higher is better) used when none is given."""
    if task == "regression":
        return "neg_root_mean_squared_error"
    return "roc_auc" if y.nunique() == 2 else "f1_weighted"


# ---------------------------------------------------------------------------
# Tuning entry point
# ---------------------------------------------------------------------------
@dataclass
class TuneResult:
    best_params: Dict[str, Any]
    best_score: float
    scoring: str
    study: optuna.Study
    final_model: Any = None  # TrainResult when train_final=True


def tune_model(
    df: pd.DataFrame,
    target: str,
    model_name: str = "lightgbm",
    task: str = "classification",
    n_trials: int = 50,
    timeout: Optional[int] = None,
    cv: int = 5,
    scoring: Optional[str] = None,
    numeric_cols: Optional[List[str]] = None,
    categorical_cols: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None,
    fixed_params: Optional[Dict[str, Any]] = None,
    storage: Optional[str] = None,
    study_name: Optional[str] = None,
    callbacks: Optional[List[Callable]] = None,
    train_final: bool = True,
    random_state: int = 42,
) -> TuneResult:
    """Cross-validated Optuna search over ``model_name``'s hyperparameters.

    Parameters
    ----------
    scoring: str | None
        Any sklearn scorer name.  Defaults to ROC-AUC (binary), weighted F1
        (multiclass) or negative RMSE (regression).
    fixed_params: dict | None
        Params applied to every trial and not tuned, e.g. ``{"class_weight": "balanced"}``.
    storage / study_name:
        Set e.g. ``storage="sqlite:///ml/artifacts/optuna.db"`` to persist and
        resume a study, or to run trials from several processes.
    callbacks:
        Passed to ``study.optimize`` - see ``ml.tracking.tracker.optuna_callback``.
    train_final: bool
        Retrain on a train/test split with the best params via ``train_model``.
    """
    X, y, numeric_cols, categorical_cols, _ = prepare_data(
        df, target, task, numeric_cols, categorical_cols, exclude
    )
    scoring = scoring or default_scoring(task, y)
    scorer = get_scorer(scoring)

    splitter_cls = StratifiedKFold if task == "classification" else KFold
    folds = list(splitter_cls(n_splits=cv, shuffle=True, random_state=random_state).split(X, y))

    def objective(trial: optuna.Trial) -> float:
        params = {**suggest_params(trial, model_name), **(fixed_params or {})}
        pipeline = build_pipeline(model_name, task, numeric_cols, categorical_cols, params)

        scores = []
        for step, (train_idx, val_idx) in enumerate(folds):
            fold_model = clone(pipeline).fit(X.iloc[train_idx], y.iloc[train_idx])
            scores.append(scorer(fold_model, X.iloc[val_idx], y.iloc[val_idx]))

            # Report the running mean so the pruner can stop bad trials after a few folds
            trial.report(float(np.mean(scores)), step)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return float(np.mean(scores))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
        storage=storage,
        study_name=study_name,
        load_if_exists=storage is not None,
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout, callbacks=callbacks)

    best_params = {**study.best_params, **(fixed_params or {})}
    if model_name == "lightgbm":
        best_params["subsample_freq"] = 1  # fixed in the search space, not a trial param

    print(f"[{model_name}] best {scoring}: {study.best_value:.4f}")
    print(f"[{model_name}] best params: {best_params}")

    final_model = None
    if train_final:
        final_model = train_model(
            df, target, model_name=model_name, task=task,
            numeric_cols=numeric_cols, categorical_cols=categorical_cols,
            params=best_params, random_state=random_state,
        )

    return TuneResult(best_params, study.best_value, scoring, study, final_model)


if __name__ == "__main__":
    from sklearn.datasets import load_breast_cancer

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    data = load_breast_cancer(as_frame=True).frame
    result = tune_model(data, target="target", model_name="lightgbm", n_trials=20, cv=3)

    # Which hyperparameters mattered most
    print(optuna.importance.get_param_importances(result.study))
