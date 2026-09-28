"""Optuna tuning for the tabular MLP: architecture, optimiser, loss function and epochs.

Trials are compared on a loss-independent validation metric (F1 / RMSE by
default), since loss values aren't comparable across loss functions.  Weak
trials are pruned epoch-by-epoch, and each trial records the epoch at which
it peaked so the final model knows how long to train.
"""
from dataclasses import asdict, replace
from typing import Any, Callable, Dict, List, Optional

import optuna
import pandas as pd

from ml.deep_learning.losses import CLASSIFICATION_LOSSES, REGRESSION_LOSSES
from ml.deep_learning.training import (
    TabularData,
    TabularResult,
    TrainConfig,
    prepare_tabular_data,
    train_on_data,
)


# ---------------------------------------------------------------------------
# Search space
# ---------------------------------------------------------------------------
def suggest_config(
    trial: optuna.Trial,
    base: TrainConfig,
    losses: Optional[List[str]] = None,
    max_epochs: int = 200,
) -> TrainConfig:
    """Sample a TrainConfig; fields not tuned here keep their ``base`` values."""
    n_layers = trial.suggest_int("n_layers", 1, 4)
    first = trial.suggest_categorical("first_layer", [64, 128, 256, 512])
    shrink = trial.suggest_categorical("shrink", [1.0, 0.5])  # constant width or funnel
    hidden_sizes = tuple(max(16, int(first * shrink ** i)) for i in range(n_layers))

    losses = losses or (CLASSIFICATION_LOSSES if base.task == "classification" else REGRESSION_LOSSES)
    loss = trial.suggest_categorical("loss", losses)

    # Loss-specific hyperparameters are only sampled when that loss is chosen
    label_smoothing, focal_gamma, huber_delta = base.label_smoothing, base.focal_gamma, base.huber_delta
    if loss == "label_smoothing":
        label_smoothing = trial.suggest_float("label_smoothing", 0.01, 0.3)
    if loss == "focal":
        focal_gamma = trial.suggest_float("focal_gamma", 0.5, 5.0)
    if loss in ("huber", "smooth_l1"):
        huber_delta = trial.suggest_float("huber_delta", 0.1, 2.0, log=True)

    optimizer = trial.suggest_categorical("optimizer", ["adam", "adamw", "sgd"])
    lr_low, lr_high = (1e-3, 1e-1) if optimizer == "sgd" else (1e-4, 1e-2)

    return replace(
        base,
        hidden_sizes=hidden_sizes,
        dropout=trial.suggest_float("dropout", 0.0, 0.5),
        activation=trial.suggest_categorical("activation", ["relu", "gelu", "silu"]),
        batch_norm=trial.suggest_categorical("batch_norm", [True, False]),
        # Epochs as a hyperparameter: it also sets the cosine / one-cycle schedule length.
        # Early stopping still cuts a trial short once the val metric stops improving.
        epochs=trial.suggest_int("epochs", 20, max_epochs, step=10),
        batch_size=trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
        lr=trial.suggest_float("lr", lr_low, lr_high, log=True),
        weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        optimizer=optimizer,
        scheduler=trial.suggest_categorical("scheduler", ["none", "cosine", "plateau", "onecycle"]),
        loss=loss,
        label_smoothing=label_smoothing,
        focal_gamma=focal_gamma,
        huber_delta=huber_delta,
    )


# ---------------------------------------------------------------------------
# Tuning entry point
# ---------------------------------------------------------------------------
def tune_tabular(
    df: pd.DataFrame,
    target: str,
    base: Optional[TrainConfig] = None,
    n_trials: int = 50,
    timeout: Optional[int] = None,
    losses: Optional[List[str]] = None,
    max_epochs: int = 200,
    storage: Optional[str] = None,
    study_name: Optional[str] = None,
    callbacks: Optional[List[Callable]] = None,
    train_final: bool = True,
    **data_kwargs,
) -> Dict[str, Any]:
    """Search MLP hyperparameters on a fixed train / val split.

    Parameters
    ----------
    base: TrainConfig
        Task, metric, device and any fields you don't want tuned.
    losses: list | None
        Restrict the loss functions compared (default: all for the task).
    callbacks:
        Passed to ``study.optimize`` - e.g. ``ml.tracking.tracker.optuna_callback``.

    Returns ``{"config", "best_score", "best_epoch", "study", "result"}``;
    ``result`` is the final retrained model when ``train_final=True``.
    """
    base = base or TrainConfig()
    data: TabularData = prepare_tabular_data(df, target, base.task, **data_kwargs)

    def objective(trial: optuna.Trial) -> float:
        config = suggest_config(trial, base, losses, max_epochs)

        def report(epoch: int, row: Dict[str, float]) -> None:
            trial.report(row[f"val_{config.metric}"], epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        result = train_on_data(data, config, on_epoch_end=report)
        trial.set_user_attr("best_epoch", result.history.best_epoch)
        trial.set_user_attr("config", {k: v if not isinstance(v, tuple) else list(v)
                                       for k, v in asdict(config).items()})
        return result.history.best_score

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=base.seed),
        # Don't judge a trial before it has had a few epochs to warm up
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
        storage=storage,
        study_name=study_name,
        load_if_exists=storage is not None,
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout, callbacks=callbacks)

    best = study.best_trial
    best_config = TrainConfig(**best.user_attrs["config"])
    best_epoch = best.user_attrs["best_epoch"]
    print(f"Best val_{base.metric}: {best.value:.4f} at epoch {best_epoch}  "
          f"loss={best_config.loss} layers={best_config.hidden_sizes} lr={best_config.lr:.2e}")

    result = None
    if train_final:
        result = train_on_data(data, best_config)
        print(f"Final test metrics: {result.test_metrics}")

    return {"config": best_config, "best_score": best.value, "best_epoch": best_epoch,
            "study": study, "result": result}


# ---------------------------------------------------------------------------
# Focused studies
# ---------------------------------------------------------------------------
def compare_losses(
    df: pd.DataFrame,
    target: str,
    config: Optional[TrainConfig] = None,
    losses: Optional[List[str]] = None,
    **data_kwargs,
) -> pd.DataFrame:
    """Train the same model once per loss function and compare on the same split."""
    config = config or TrainConfig()
    data = prepare_tabular_data(df, target, config.task, **data_kwargs)
    losses = losses or (CLASSIFICATION_LOSSES if config.task == "classification" else REGRESSION_LOSSES)

    rows = []
    for loss in losses:
        result: TabularResult = train_on_data(data, replace(config, loss=loss))
        rows.append({
            "loss": loss,
            "best_epoch": result.history.best_epoch,
            "epochs_run": len(result.history.epochs),
            f"val_{config.metric}": round(result.history.best_score, 4),
            **{f"test_{k}": v for k, v in result.test_metrics.items()},
        })
    return pd.DataFrame(rows).sort_values(f"val_{config.metric}", ascending=False).reset_index(drop=True)


def epoch_curve(result: TabularResult) -> pd.DataFrame:
    """Per-epoch history - plot train_loss / val_loss to spot over- or under-fitting."""
    return pd.DataFrame(result.history.epochs).set_index("epoch")


if __name__ == "__main__":
    from sklearn.datasets import load_breast_cancer

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    data = load_breast_cancer(as_frame=True).frame

    print(compare_losses(data, "target", TrainConfig(epochs=60, device="cpu")), "\n")

    tuned = tune_tabular(data, "target", base=TrainConfig(device="cpu"), n_trials=15, max_epochs=100)
    print(epoch_curve(tuned["result"]).tail())
