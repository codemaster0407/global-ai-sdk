"""Anomaly detection for time series: statistical, Isolation Forest and forecast residuals."""
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Statistical detectors (no training needed)
# ---------------------------------------------------------------------------
def rolling_zscore(series: pd.Series, window: int = 30, threshold: float = 3.0) -> pd.DataFrame:
    """Flag points more than ``threshold`` std devs from the trailing rolling mean."""
    past = series.shift(1).rolling(window, min_periods=window // 2)
    z = (series - past.mean()) / past.std()
    return pd.DataFrame({"value": series, "score": z, "is_anomaly": z.abs() > threshold})


def iqr_outliers(series: pd.Series, k: float = 1.5) -> pd.DataFrame:
    """Flag points outside ``[Q1 - k*IQR, Q3 + k*IQR]`` over the whole series."""
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - k * iqr, q3 + k * iqr
    return pd.DataFrame({"value": series, "is_anomaly": (series < lower) | (series > upper)})


# ---------------------------------------------------------------------------
# Isolation Forest (multivariate, trainable)
# ---------------------------------------------------------------------------
class IsolationForestDetector:
    """Train once on normal-ish data, then score new rows."""

    def __init__(self, feature_cols: List[str], contamination: float | str = "auto", random_state: int = 42):
        self.feature_cols = feature_cols
        self.pipeline = Pipeline([
            ("scale", StandardScaler()),
            ("model", IsolationForest(n_estimators=300, contamination=contamination, random_state=random_state)),
        ])

    def fit(self, df: pd.DataFrame) -> "IsolationForestDetector":
        self.pipeline.fit(df[self.feature_cols])
        return self

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add ``anomaly_score`` (higher = more anomalous) and ``is_anomaly`` columns."""
        X = df[self.feature_cols]
        scores = -self.pipeline.decision_function(X)
        return df.assign(anomaly_score=scores, is_anomaly=self.pipeline.predict(X) == -1)

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "IsolationForestDetector":
        return joblib.load(path)


# ---------------------------------------------------------------------------
# Forecast residual detector
# ---------------------------------------------------------------------------
def residual_anomalies(
    actual: pd.Series,
    predicted: pd.Series,
    threshold: float = 3.0,
    residual_std: Optional[float] = None,
) -> pd.DataFrame:
    """Flag points whose forecast error is large relative to typical error.

    Pair with ``ml.time_series.forecasting`` - pass ``residual_std`` measured on
    a validation window, otherwise it's estimated from these residuals.
    """
    residual = actual - predicted
    std = residual_std or residual.std()
    score = residual / std
    return pd.DataFrame({
        "actual": actual,
        "predicted": predicted,
        "score": score,
        "is_anomaly": score.abs() > threshold,
    })


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    idx = pd.date_range("2025-01-01", periods=200, freq="h")
    values = pd.Series(50 + rng.normal(0, 2, 200), index=idx)
    values.iloc[[40, 120, 170]] += [25, -20, 30]  # injected anomalies

    print("Rolling z-score:\n", rolling_zscore(values).query("is_anomaly"))

    frame = pd.DataFrame({"value": values, "diff": values.diff().fillna(0)})
    detector = IsolationForestDetector(["value", "diff"], contamination=0.02).fit(frame)
    print("Isolation Forest:\n", detector.detect(frame).query("is_anomaly"))
