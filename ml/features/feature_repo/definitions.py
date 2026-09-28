"""Example Feast definitions: a customer entity and its daily stats."""
import os
from datetime import timedelta

from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int64

CUSTOMER_STATS_PATH = os.path.join(os.path.dirname(__file__), "data", "customer_stats.parquet")

customer = Entity(name="customer", join_keys=["customer_id"])

customer_stats_source = FileSource(
    path=CUSTOMER_STATS_PATH,
    timestamp_field="event_timestamp",
    created_timestamp_column="created",
)

customer_stats_fv = FeatureView(
    name="customer_stats",
    entities=[customer],
    ttl=timedelta(days=2),
    schema=[
        Field(name="avg_order_value", dtype=Float32),
        Field(name="orders_30d", dtype=Int64),
        Field(name="days_since_last_order", dtype=Int64),
    ],
    source=customer_stats_source,
    online=True,
)
