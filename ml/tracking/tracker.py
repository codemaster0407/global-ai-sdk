"""One experiment-tracking interface over MLflow and Weights & Biases.

    tracker = get_tracker("mlflow", experiment="churn")     # or "wandb", or "none"
    with tracker.run("lgbm-baseline", config={"model": "lightgbm"}):
        tracker.log_metrics({"auc": 0.91})
        tracker.log_model(pipeline, "model")

Setup:
    MLflow - runs locally by default; set MLFLOW_TRACKING_URI for a server.
             UI: ``mlflow ui --backend-store-uri sqlite:///ml/artifacts/mlflow.db``
    W&B    - ``wandb login`` or set WANDB_API_KEY; use mode="offline" to log locally.
"""
import os
import tempfile
from contextlib import contextmanager
from typing import Any, Dict, Optional

import joblib

DEFAULT_MLFLOW_URI = "sqlite:///ml/artifacts/mlflow.db"
DEFAULT_MLFLOW_ARTIFACTS = "ml/artifacts/mlruns"


class BaseTracker:
    """No-op tracker; also the interface the real trackers implement."""

    @contextmanager
    def run(self, name: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        yield self

    def log_params(self, params: Dict[str, Any]) -> None: ...
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None: ...
    def log_artifact(self, path: str) -> None: ...
    def log_model(self, model: Any, name: str = "model") -> None: ...
    def set_tags(self, tags: Dict[str, str]) -> None: ...


# ---------------------------------------------------------------------------
# MLflow
# ---------------------------------------------------------------------------
class MLflowTracker(BaseTracker):
    def __init__(self, experiment: str = "default", tracking_uri: Optional[str] = None):
        import mlflow

        self.mlflow = mlflow
        uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", DEFAULT_MLFLOW_URI)
        mlflow.set_tracking_uri(uri)

        # For the local default, keep run artifacts under ml/artifacts instead of ./mlruns
        if uri == DEFAULT_MLFLOW_URI and mlflow.get_experiment_by_name(experiment) is None:
            mlflow.create_experiment(experiment, artifact_location=os.path.abspath(DEFAULT_MLFLOW_ARTIFACTS))
        mlflow.set_experiment(experiment)

    @contextmanager
    def run(self, name: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        # nested=True lets you open child runs (e.g. one per tuning trial) inside a parent run
        with self.mlflow.start_run(run_name=name, nested=self.mlflow.active_run() is not None):
            if config:
                self.log_params(config)
            yield self

    def log_params(self, params: Dict[str, Any]) -> None:
        # MLflow params are strings with a length limit, so stringify complex values
        self.mlflow.log_params({k: str(v)[:500] for k, v in params.items()})

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        self.mlflow.log_metrics({k: float(v) for k, v in metrics.items()}, step=step)

    def log_artifact(self, path: str) -> None:
        self.mlflow.log_artifact(path)

    def log_model(self, model: Any, name: str = "model") -> None:
        if _is_torch_module(model):
            import mlflow.pytorch
            mlflow.pytorch.log_model(model, name=name)
        else:
            import mlflow.sklearn
            # cloudpickle handles LightGBM / XGBoost estimators inside sklearn pipelines
            mlflow.sklearn.log_model(model, name=name, serialization_format="cloudpickle")

    def set_tags(self, tags: Dict[str, str]) -> None:
        self.mlflow.set_tags(tags)


# ---------------------------------------------------------------------------
# Weights & Biases
# ---------------------------------------------------------------------------
class WandbTracker(BaseTracker):
    def __init__(self, project: str = "global-ai-sdk", entity: Optional[str] = None, mode: Optional[str] = None):
        import wandb

        self.wandb = wandb
        self.project = project
        self.entity = entity
        self.mode = mode or os.getenv("WANDB_MODE")  # "online" | "offline" | "disabled"
        self._run = None

    @contextmanager
    def run(self, name: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        self._run = self.wandb.init(
            project=self.project, entity=self.entity, name=name,
            config=config, mode=self.mode, reinit="finish_previous",
        )
        try:
            yield self
        finally:
            self._run.finish()
            self._run = None

    def log_params(self, params: Dict[str, Any]) -> None:
        self._run.config.update(params, allow_val_change=True)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        self._run.log(metrics, step=step)

    def log_artifact(self, path: str) -> None:
        artifact = self.wandb.Artifact(os.path.basename(path).replace(".", "_"), type="file")
        artifact.add_file(path)
        self._run.log_artifact(artifact)

    def log_model(self, model: Any, name: str = "model") -> None:
        with tempfile.TemporaryDirectory() as tmp:
            if _is_torch_module(model):
                import torch
                path = os.path.join(tmp, f"{name}.pt")
                torch.save(model.state_dict(), path)
            else:
                path = os.path.join(tmp, f"{name}.joblib")
                joblib.dump(model, path)
            artifact = self.wandb.Artifact(name, type="model")
            artifact.add_file(path)
            self._run.log_artifact(artifact)

    def set_tags(self, tags: Dict[str, str]) -> None:
        self._run.tags = tuple(self._run.tags or ()) + tuple(f"{k}:{v}" for k, v in tags.items())


def _is_torch_module(model: Any) -> bool:
    try:
        import torch
        return isinstance(model, torch.nn.Module)
    except ImportError:
        return False


def get_tracker(backend: str = "mlflow", **kwargs) -> BaseTracker:
    """``backend``: ``mlflow``, ``wandb`` or ``none``."""
    if backend == "mlflow":
        return MLflowTracker(**kwargs)
    if backend == "wandb":
        return WandbTracker(**kwargs)
    if backend == "none":
        return BaseTracker()
    raise ValueError(f"Unknown tracking backend '{backend}'")


# ---------------------------------------------------------------------------
# Integrations
# ---------------------------------------------------------------------------
def train_with_tracking(tracker: BaseTracker, run_name: Optional[str] = None, **train_kwargs):
    """Run ``ml.classical.training.train_model`` and log config, metrics and model."""
    from ml.classical.training import train_model

    params = train_kwargs.pop("params", None) or {}
    config = {**{k: v for k, v in train_kwargs.items() if k != "df"}, **params}

    with tracker.run(run_name, config=config):
        result = train_model(**train_kwargs, params=params)
        tracker.log_metrics(result.metrics)
        tracker.log_params({
            "n_train": result.metadata["n_train"],
            "n_features": len(result.metadata["numeric_cols"]) + len(result.metadata["categorical_cols"]),
        })
        tracker.log_model(result.model, "model")
    return result


def optuna_callback(tracker: BaseTracker, log_each_trial_as_run: bool = False):
    """Optuna callback that logs every trial to the tracker.

    By default trials are logged as steps of the current run (call inside
    ``tracker.run(...)``).  With ``log_each_trial_as_run=True`` each trial
    becomes its own (nested) run with its params, which is easier to compare
    in the MLflow / W&B UI.
    """
    def callback(study, trial):
        if trial.value is None:  # pruned or failed
            return
        if log_each_trial_as_run:
            with tracker.run(f"trial-{trial.number}", config=trial.params):
                tracker.log_metrics({"score": trial.value})
        else:
            tracker.log_metrics({"trial_score": trial.value, "best_score": study.best_value}, step=trial.number)

    return callback


if __name__ == "__main__":
    from sklearn.datasets import load_breast_cancer

    data = load_breast_cancer(as_frame=True).frame
    tracker = get_tracker("mlflow", experiment="breast-cancer")
    train_with_tracking(
        tracker, run_name="lgbm-baseline",
        df=data, target="target", model_name="lightgbm",
        params={"num_leaves": 31, "learning_rate": 0.05},
    )
    print("Logged to MLflow. View with: mlflow ui --backend-store-uri", DEFAULT_MLFLOW_URI)
