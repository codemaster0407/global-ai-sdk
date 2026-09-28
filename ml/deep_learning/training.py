"""PyTorch training loop for tabular models: epochs, early stopping, schedulers.

    config = TrainConfig(task="classification", loss="focal", epochs=100)
    result = train_tabular(df, target="label", config=config)
"""
import random
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset

from ml.classical.training import prepare_data
from ml.deep_learning.losses import get_loss
from ml.deep_learning.models import TabularMLP
from ml.features.pipelines import build_preprocessor


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class TrainConfig:
    task: str = "classification"            # or "regression"
    # architecture
    hidden_sizes: Sequence[int] = (256, 128)
    dropout: float = 0.2
    activation: str = "relu"
    batch_norm: bool = True
    # optimisation
    epochs: int = 100
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: str = "adamw"                 # adam | adamw | sgd
    scheduler: str = "cosine"                # none | cosine | plateau | onecycle
    grad_clip: Optional[float] = 1.0
    # loss (see ml.deep_learning.losses)
    loss: str = "cross_entropy"
    label_smoothing: float = 0.1
    focal_gamma: float = 2.0
    huber_delta: float = 1.0
    # early stopping on the validation metric (higher is better)
    metric: Optional[str] = None             # accuracy | f1_macro | roc_auc | neg_rmse | neg_mae
    early_stopping_patience: Optional[int] = 10
    # misc
    device: str = "auto"
    seed: int = 42

    def __post_init__(self):
        if self.metric is None:
            self.metric = "f1_macro" if self.task == "classification" else "neg_rmse"
        if self.task == "regression" and self.loss == "cross_entropy":
            self.loss = "mse"


def get_device(preference: str = "auto") -> torch.device:
    if preference != "auto":
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
@dataclass
class TabularData:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    preprocessor: Any
    task: str
    classes: Optional[List[Any]] = None      # original labels, index = encoded class id
    y_mean: float = 0.0                      # regression targets are standardised for training
    y_std: float = 1.0

    def unscale(self, values: np.ndarray) -> np.ndarray:
        """Map regression outputs / targets back to the original units."""
        return values * self.y_std + self.y_mean if self.task == "regression" else values

    @property
    def n_features(self) -> int:
        return self.X_train.shape[1]

    @property
    def n_outputs(self) -> int:
        return len(self.classes) if self.task == "classification" else 1

    def class_weights(self) -> Optional[List[float]]:
        if self.task != "classification":
            return None
        return compute_class_weight("balanced", classes=np.arange(self.n_outputs), y=self.y_train).tolist()


def prepare_tabular_data(
    df: pd.DataFrame,
    target: str,
    task: str = "classification",
    val_size: float = 0.15,
    test_size: float = 0.15,
    exclude: Optional[List[str]] = None,
    random_state: int = 42,
) -> TabularData:
    """Split train / val / test, fit the preprocessor on train only, return float32 arrays."""
    X, y, numeric_cols, categorical_cols, _ = prepare_data(df, target, task, exclude=exclude)

    classes = None
    if task == "classification":
        # Map any label set (strings, {1, 2}, ...) to 0..K-1 as CrossEntropyLoss expects
        classes = sorted(df[target].unique().tolist())
        y = df[target].map({c: i for i, c in enumerate(classes)})

    stratify = y if task == "classification" else None
    X_rest, X_test, y_rest, y_test = train_test_split(
        X, y, test_size=test_size, stratify=stratify, random_state=random_state
    )
    stratify = y_rest if task == "classification" else None
    X_train, X_val, y_train, y_val = train_test_split(
        X_rest, y_rest, test_size=val_size / (1 - test_size), stratify=stratify, random_state=random_state
    )

    # Neural nets need scaled inputs - and, for regression, a scaled target
    preprocessor = build_preprocessor(numeric_cols, categorical_cols, scale=True)
    to_np = lambda frame: np.asarray(frame, dtype=np.float32)

    y_mean, y_std = 0.0, 1.0
    if task == "classification":
        to_y = lambda s: np.asarray(s, dtype=np.int64)
    else:
        y_mean, y_std = float(y_train.mean()), float(y_train.std()) or 1.0
        to_y = lambda s: ((np.asarray(s, dtype=np.float64) - y_mean) / y_std).astype(np.float32)

    return TabularData(
        X_train=to_np(preprocessor.fit_transform(X_train)),
        y_train=to_y(y_train),
        X_val=to_np(preprocessor.transform(X_val)),
        y_val=to_y(y_val),
        X_test=to_np(preprocessor.transform(X_test)),
        y_test=to_y(y_test),
        preprocessor=preprocessor,
        task=task,
        classes=classes,
        y_mean=y_mean,
        y_std=y_std,
    )


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    # drop_last avoids a size-1 final batch, which BatchNorm can't train on
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      drop_last=shuffle and len(dataset) > batch_size)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def score(metric: str, y_true: np.ndarray, outputs: np.ndarray) -> float:
    """Validation metric, higher is better.  ``outputs`` are probabilities or predictions."""
    if metric == "accuracy":
        return accuracy_score(y_true, outputs.argmax(1))
    if metric == "f1_macro":
        return f1_score(y_true, outputs.argmax(1), average="macro")
    if metric == "roc_auc":
        if outputs.shape[1] == 2:
            return roc_auc_score(y_true, outputs[:, 1])
        return roc_auc_score(y_true, outputs, multi_class="ovr")
    if metric == "neg_rmse":
        return -float(np.sqrt(mean_squared_error(y_true, outputs)))
    if metric == "neg_mae":
        return -mean_absolute_error(y_true, outputs)
    raise ValueError(f"Unknown metric '{metric}'")


# ---------------------------------------------------------------------------
# Optimiser / scheduler
# ---------------------------------------------------------------------------
def build_optimizer(model: torch.nn.Module, config: TrainConfig) -> torch.optim.Optimizer:
    params = model.parameters()
    if config.optimizer == "adam":
        return torch.optim.Adam(params, lr=config.lr, weight_decay=config.weight_decay)
    if config.optimizer == "adamw":
        return torch.optim.AdamW(params, lr=config.lr, weight_decay=config.weight_decay)
    if config.optimizer == "sgd":
        return torch.optim.SGD(params, lr=config.lr, momentum=0.9, nesterov=True, weight_decay=config.weight_decay)
    raise ValueError(f"Unknown optimizer '{config.optimizer}'")


def build_scheduler(optimizer, config: TrainConfig, steps_per_epoch: int):
    """Returns (scheduler, step_per_batch)."""
    if config.scheduler == "none":
        return None, False
    if config.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs), False
    if config.scheduler == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3), False
    if config.scheduler == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=config.lr, epochs=config.epochs, steps_per_epoch=steps_per_epoch
        ), True
    raise ValueError(f"Unknown scheduler '{config.scheduler}'")


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
@dataclass
class History:
    epochs: List[Dict[str, float]] = field(default_factory=list)
    best_epoch: int = 0
    best_score: float = float("-inf")
    stopped_early: bool = False


@torch.no_grad()
def predict(model: torch.nn.Module, X: np.ndarray, task: str, device: torch.device, batch_size: int = 4096) -> np.ndarray:
    """Class probabilities (classification) or predictions (regression)."""
    model.eval()
    outputs = []
    for start in range(0, len(X), batch_size):
        batch = torch.from_numpy(X[start:start + batch_size]).to(device)
        out = model(batch)
        outputs.append((out.softmax(-1) if task == "classification" else out).cpu().numpy())
    return np.concatenate(outputs)


@torch.no_grad()
def compute_loss(model, X: np.ndarray, y: np.ndarray, loss_fn, device, batch_size: int = 4096) -> float:
    """Mean loss over a dataset, computed on raw model outputs (logits)."""
    model.eval()
    total = 0.0
    for start in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[start:start + batch_size]).to(device)
        yb = torch.from_numpy(y[start:start + batch_size]).to(device)
        total += loss_fn(model(xb), yb).item() * len(xb)
    return total / len(X)


def fit(
    model: torch.nn.Module,
    data: TabularData,
    config: TrainConfig,
    on_epoch_end: Optional[Callable[[int, Dict[str, float]], None]] = None,
) -> History:
    """Train with early stopping on ``config.metric``; restores the best weights.

    ``on_epoch_end(epoch, row)`` runs after every epoch - use it for logging
    (``tracker.log_metrics``) or Optuna pruning (raise ``optuna.TrialPruned``).
    """
    set_seed(config.seed)
    device = get_device(config.device)
    model.to(device)

    loss_fn = get_loss(
        config.loss, class_weights=data.class_weights(),
        label_smoothing=config.label_smoothing, focal_gamma=config.focal_gamma,
        huber_delta=config.huber_delta,
    ).to(device)

    train_loader = make_loader(data.X_train, data.y_train, config.batch_size, shuffle=True)
    optimizer = build_optimizer(model, config)
    scheduler, step_per_batch = build_scheduler(optimizer, config, len(train_loader))

    history = History()
    best_state = None
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, config.epochs + 1):
        model.train()
        total_loss, n = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            if config.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            if scheduler is not None and step_per_batch:
                scheduler.step()
            total_loss += loss.item() * len(xb)
            n += len(xb)

        val_outputs = predict(model, data.X_val, config.task, device)
        val_loss = compute_loss(model, data.X_val, data.y_val, loss_fn, device)
        val_score = score(config.metric, data.unscale(data.y_val), data.unscale(val_outputs))

        if scheduler is not None and not step_per_batch:
            scheduler.step(val_score) if config.scheduler == "plateau" else scheduler.step()

        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(n, 1),
            "val_loss": val_loss,
            f"val_{config.metric}": val_score,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.epochs.append(row)

        # Metrics like F1 / accuracy plateau on small val sets; break ties with val loss
        improved = val_score > history.best_score or (
            val_score == history.best_score and val_loss < best_val_loss
        )
        if improved:
            history.best_score, history.best_epoch, best_val_loss = val_score, epoch, val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if on_epoch_end is not None:
            on_epoch_end(epoch, row)

        if config.early_stopping_patience and epochs_without_improvement >= config.early_stopping_patience:
            history.stopped_early = True
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return history


def evaluate(model: torch.nn.Module, data: TabularData, config: TrainConfig) -> Dict[str, float]:
    """Test-set metrics, in the target's original units for regression."""
    outputs = data.unscale(predict(model, data.X_test, config.task, get_device(config.device)))
    y = data.unscale(data.y_test)
    if config.task == "classification":
        metrics = {
            "accuracy": accuracy_score(y, outputs.argmax(1)),
            "f1_macro": f1_score(y, outputs.argmax(1), average="macro"),
            "roc_auc": score("roc_auc", y, outputs),
        }
    else:
        metrics = {
            "mae": mean_absolute_error(y, outputs),
            "rmse": float(np.sqrt(mean_squared_error(y, outputs))),
            "r2": r2_score(y, outputs),
        }
    return {k: round(float(v), 4) for k, v in metrics.items()}


# ---------------------------------------------------------------------------
# End-to-end entry point
# ---------------------------------------------------------------------------
@dataclass
class TabularResult:
    model: torch.nn.Module
    data: TabularData
    config: TrainConfig
    history: History
    test_metrics: Dict[str, float]


def build_tabular_model(data: TabularData, config: TrainConfig) -> TabularMLP:
    return TabularMLP(
        data.n_features, data.n_outputs, hidden_sizes=config.hidden_sizes,
        dropout=config.dropout, activation=config.activation, batch_norm=config.batch_norm,
    )


def train_on_data(
    data: TabularData,
    config: TrainConfig,
    on_epoch_end: Optional[Callable[[int, Dict[str, float]], None]] = None,
) -> TabularResult:
    set_seed(config.seed)
    model = build_tabular_model(data, config)
    history = fit(model, data, config, on_epoch_end)
    test_metrics = evaluate(model, data, config)
    return TabularResult(model, data, config, history, test_metrics)


def train_tabular(
    df: pd.DataFrame,
    target: str,
    config: Optional[TrainConfig] = None,
    tracker=None,
    run_name: Optional[str] = None,
    output_path: Optional[str] = None,
    **data_kwargs,
) -> TabularResult:
    """Prepare data, train an MLP and report test metrics.

    Pass a tracker from ``ml.tracking.tracker`` to log config, per-epoch
    metrics and the final model to MLflow / W&B.
    """
    config = config or TrainConfig()
    data = prepare_tabular_data(df, target, config.task, **data_kwargs)

    if tracker is None:
        result = train_on_data(data, config)
    else:
        with tracker.run(run_name, config=asdict(config)):
            log_epoch = lambda epoch, row: tracker.log_metrics(
                {k: v for k, v in row.items() if k != "epoch"}, step=epoch
            )
            result = train_on_data(data, config, on_epoch_end=log_epoch)
            tracker.log_metrics({f"test_{k}": v for k, v in result.test_metrics.items()})
            tracker.log_model(result.model, "model")

    h = result.history
    print(f"[mlp/{config.loss}] best epoch {h.best_epoch}/{len(h.epochs)} "
          f"val_{config.metric}={h.best_score:.4f}  test={result.test_metrics}")

    if output_path:
        save_tabular(result, output_path)
    return result


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def save_tabular(result: TabularResult, path: str) -> None:
    """Save weights, config, preprocessor and label mapping in one file."""
    joblib.dump({
        "state_dict": {k: v.cpu() for k, v in result.model.state_dict().items()},
        "config": asdict(result.config),
        "preprocessor": result.data.preprocessor,
        "classes": result.data.classes,
        "y_mean": result.data.y_mean,
        "y_std": result.data.y_std,
        "n_features": result.data.n_features,
        "n_outputs": result.data.n_outputs,
    }, path)
    print(f"Model saved to {path}")


class TabularPredictor:
    """Load a model saved by ``save_tabular`` and predict on raw DataFrames."""

    def __init__(self, path: str):
        bundle = joblib.load(path)
        self.config = TrainConfig(**bundle["config"])
        self.preprocessor = bundle["preprocessor"]
        self.classes = bundle["classes"]
        self.y_mean, self.y_std = bundle["y_mean"], bundle["y_std"]
        self.device = get_device(self.config.device)
        self.model = TabularMLP(
            bundle["n_features"], bundle["n_outputs"], hidden_sizes=self.config.hidden_sizes,
            dropout=self.config.dropout, activation=self.config.activation, batch_norm=self.config.batch_norm,
        )
        self.model.load_state_dict(bundle["state_dict"])
        self.model.to(self.device)

    def _outputs(self, df: pd.DataFrame) -> np.ndarray:
        X = np.asarray(self.preprocessor.transform(df), dtype=np.float32)
        return predict(self.model, X, self.config.task, self.device)

    def predict(self, df: pd.DataFrame) -> list:
        outputs = self._outputs(df)
        if self.config.task == "classification":
            return [self.classes[i] for i in outputs.argmax(1)]
        return (outputs * self.y_std + self.y_mean).tolist()

    def predict_proba(self, df: pd.DataFrame) -> List[Dict[str, float]]:
        if self.config.task != "classification":
            raise ValueError("predict_proba is only available for classification models")
        outputs = self._outputs(df)
        return [{str(c): round(float(p), 4) for c, p in zip(self.classes, row)} for row in outputs]


if __name__ == "__main__":
    from sklearn.datasets import load_breast_cancer

    data = load_breast_cancer(as_frame=True).frame
    result = train_tabular(
        data, target="target",
        config=TrainConfig(task="classification", loss="focal", epochs=50, device="cpu"),
        output_path="ml/artifacts/breast_cancer_mlp.joblib",
    )

    predictor = TabularPredictor("ml/artifacts/breast_cancer_mlp.joblib")
    print(predictor.predict_proba(data.drop(columns="target").head(3)))
