from datetime import timedelta

import ibis

from fraudsys import data
from fraudsys.features.store.client import (
    Aggregate,
    BatchSource,
    Entity,
    FeatureService,
    FeatureView,
    Field,
)

transactions_source = BatchSource(
    name="transactions_source",
    description="Source definition for transactions data.",
    loader=data.ParquetLoader(
        path="data/training/inputs_train.parquet", dataframe_type="pandas"
    ),
    timestamp_field="transaction_datetime",
)

customer_entity = Entity(
    name="customer", description="entity for customers", join_keys=["customer_id"]
)

customer_stats_fv = FeatureView(
    name="customer_stats_fv",
    description="Feature view for customer stats.",
    source=transactions_source,
    entity=customer_entity,
    aggregation_interval=timedelta(hours=1),  # compute hourly
    timestamp_field="transaction_datetime",
    features=[
        Aggregate(
            field=Field(name="amount_usd", dtype=float),
            function="count",
            time_window=timedelta(days=7),
        ),
        Aggregate(
            field=Field(name="amount_usd", dtype=float),
            function="count",
            time_window=timedelta(days=1),
        ),
        Aggregate(
            field=Field(name="amount_usd", dtype=float),
            function="mean",
            time_window=timedelta(days=1),
        ),
        Aggregate(
            field=Field(name="amount_usd", dtype=float),
            function="mean",
            time_window=timedelta(days=7),
        ),
    ],
)


if __name__ == "__main__":
    fraud_fs = FeatureService(
        name="fraud_feature_service_v1",
        description="Fraud detection system feature service",
        feature_views=[customer_stats_fv],
        tags={"version": "v1"},
    )

    ibis.options.interactive = True
    entity_df = ibis.read_parquet("data/training/inputs_train.parquet")

    fraud_fs.get_historical_features(entity_df=entity_df)
