"""Forecast a single time series with lag features and a gradient-boosted model."""
from dataclasses import dataclass
from typing import Dict, List, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

from ml.classical.training import build_model


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
def make_features(
    series: pd.Series,
    lags: Sequence[int] = (1, 7, 14),
    windows: Sequence[int] = (7, 28),
) -> pd.DataFrame:
    """Build lag, rolling and calendar features from a series with a DatetimeIndex.

    Rolling stats are shifted by one step so a row never sees its own value.
    """
    df = pd.DataFrame({"y": series})
    for lag in lags:
        df[f"lag_{lag}"] = series.shift(lag)
    for w in windows:
        shifted = series.shift(1)
        df[f"roll_mean_{w}"] = shifted.rolling(w).mean()
        df[f"roll_std_{w}"] = shifted.rolling(w).std()

    idx = series.index
    df["dayofweek"] = idx.dayofweek
    df["month"] = idx.month
    df["dayofyear"] = idx.dayofyear
    return df


# ---------------------------------------------------------------------------
# Forecaster
# ---------------------------------------------------------------------------
@dataclass
class Forecaster:
    model: object
    lags: Sequence[int]
    windows: Sequence[int]
    freq: str
    feature_cols: List[str]

    def forecast(self, history: pd.Series, horizon: int) -> pd.Series:
        """Recursive multi-step forecast: each prediction feeds the next step's lags."""
        series = history.copy()
        for _ in range(horizon):
            next_ts = series.index[-1] + pd.tseries.frequencies.to_offset(self.freq)
            extended = pd.concat([series, pd.Series([np.nan], index=[next_ts])])
            row = make_features(extended, self.lags, self.windows).iloc[[-1]][self.feature_cols]
            series.loc[next_ts] = float(self.model.predict(row)[0])
        return series.iloc[-horizon:]

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "Forecaster":
        return joblib.load(path)


def train_forecaster(
    df: pd.DataFrame,
    time_col: str,
    target: str,
    model_name: str = "lightgbm",
    lags: Sequence[int] = (1, 7, 14),
    windows: Sequence[int] = (7, 28),
    freq: str | None = None,
) -> Forecaster:
    """Fit a forecaster on ``df[target]`` indexed by ``df[time_col]``."""
    series = df.set_index(pd.to_datetime(df[time_col]))[target].sort_index()
    freq = freq or pd.infer_freq(series.index)
    if freq is None:
        raise ValueError("Could not infer series frequency; pass freq explicitly (e.g. 'D')")

    feats = make_features(series, lags, windows).dropna()
    feature_cols = [c for c in feats.columns if c != "y"]

    model = build_model(model_name, task="regression")
    model.fit(feats[feature_cols], feats["y"])
    return Forecaster(model, lags, windows, freq, feature_cols)


# ---------------------------------------------------------------------------
# Backtesting
# ---------------------------------------------------------------------------
def backtest(
    df: pd.DataFrame,
    time_col: str,
    target: str,
    horizon: int = 14,
    n_splits: int = 3,
    **forecaster_kwargs,
) -> Dict[str, float]:
    """Walk-forward evaluation: train on each prefix, forecast the next ``horizon`` steps."""
    df = df.sort_values(time_col).reset_index(drop=True)
    splitter = TimeSeriesSplit(n_splits=n_splits, test_size=horizon)

    maes, rmses, mapes = [], [], []
    for train_idx, test_idx in splitter.split(df):
        train, test = df.iloc[train_idx], df.iloc[test_idx]
        forecaster = train_forecaster(train, time_col, target, **forecaster_kwargs)

        history = train.set_index(pd.to_datetime(train[time_col]))[target]
        preds = forecaster.forecast(history, len(test)).values
        actual = test[target].values

        maes.append(mean_absolute_error(actual, preds))
        rmses.append(np.sqrt(mean_squared_error(actual, preds)))
        nonzero = actual != 0
        mapes.append(np.mean(np.abs((actual[nonzero] - preds[nonzero]) / actual[nonzero])) * 100)

    return {
        "mae": round(float(np.mean(maes)), 4),
        "rmse": round(float(np.mean(rmses)), 4),
        "mape": round(float(np.mean(mapes)), 2),
    }


def make_sample_series(days: int = 365) -> pd.DataFrame:
    """Daily series with trend, weekly seasonality and noise."""
    rng = np.random.default_rng(0)
    dates = pd.date_range("2025-01-01", periods=days, freq="D")
    t = np.arange(days)
    values = 100 + 0.1 * t + 10 * np.sin(2 * np.pi * t / 7) + rng.normal(0, 3, days)
    return pd.DataFrame({"date": dates, "sales": values})


if __name__ == "__main__":
    data = make_sample_series()
    print("Backtest:", backtest(data, "date", "sales", horizon=14))

    forecaster = train_forecaster(data, "date", "sales")
    history = data.set_index("date")["sales"]
    print(forecaster.forecast(history, horizon=7))
