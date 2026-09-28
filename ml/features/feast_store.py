"""Helpers around a Feast feature store (definitions live in ``feature_repo``).

One-time setup:
    pip install feast
    python -m ml.features.feast_store       # writes sample data, applies, materializes
"""
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from feast import FeatureStore

REPO_PATH = os.path.join(os.path.dirname(__file__), "feature_repo")


def get_store(repo_path: str = REPO_PATH) -> FeatureStore:
    return FeatureStore(repo_path=repo_path)


def apply_definitions(store: FeatureStore) -> None:
    """Register the entities and feature views from ``feature_repo/definitions.py``.

    Equivalent to running ``feast apply`` inside the repo directory.
    """
    from ml.features.feature_repo.definitions import customer, customer_stats_fv

    store.apply([customer, customer_stats_fv])


def get_training_data(
    entity_df: pd.DataFrame,
    features: List[str],
    store: FeatureStore | None = None,
) -> pd.DataFrame:
    """Point-in-time correct join of features onto ``entity_df``.

    ``entity_df`` needs the entity join keys plus an ``event_timestamp``
    column; each row gets the feature values as they were at that time.
    """
    store = store or get_store()
    return store.get_historical_features(entity_df=entity_df, features=features).to_df()


def get_online_features(
    entity_rows: List[Dict[str, Any]],
    features: List[str],
    store: FeatureStore | None = None,
) -> pd.DataFrame:
    """Fetch the latest feature values for low-latency inference."""
    store = store or get_store()
    return store.get_online_features(features=features, entity_rows=entity_rows).to_df()


def materialize(store: FeatureStore | None = None, end_date: datetime | None = None) -> None:
    """Load new offline feature values into the online store."""
    store = store or get_store()
    store.materialize_incremental(end_date=end_date or datetime.now())


def write_sample_data(path: str, n_customers: int = 100, days: int = 30) -> pd.DataFrame:
    """Generate a daily customer stats parquet file for the example feature view."""
    rng = np.random.default_rng(0)
    now = pd.Timestamp.now(tz="UTC").floor("D")
    rows = [
        {
            "customer_id": cid,
            "event_timestamp": now - pd.Timedelta(days=d),
            "created": now,
            "avg_order_value": float(rng.gamma(2.0, 30.0)),
            "orders_30d": int(rng.poisson(4)),
            "days_since_last_order": int(rng.integers(0, 60)),
        }
        for cid in range(1, n_customers + 1)
        for d in range(days)
    ]
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)
    return df


if __name__ == "__main__":
    from ml.features.feature_repo.definitions import CUSTOMER_STATS_PATH

    write_sample_data(CUSTOMER_STATS_PATH)
    store = get_store()
    apply_definitions(store)
    materialize(store)

    features = [
        "customer_stats:avg_order_value",
        "customer_stats:orders_30d",
        "customer_stats:days_since_last_order",
    ]

    entity_df = pd.DataFrame({
        "customer_id": [1, 2, 3],
        "event_timestamp": [pd.Timestamp.now(tz="UTC") - timedelta(days=5)] * 3,
    })
    print(get_training_data(entity_df, features, store))
    print(get_online_features([{"customer_id": 1}, {"customer_id": 2}], features, store))
