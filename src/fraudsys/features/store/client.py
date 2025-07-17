# classes (logic) for custom feature store
from fraudsys import data
from fraudsys.features.store import definitions as defs

customer_stats_source = defs.BatchSource(
    name="customer_stats_source",
    description="Source definition for customer stats.",
    loader=data.ParquetLoader(
        path="data/features/customer_stats_7d.parquet", dataframe_type="pandas"
    ),
    timestamp_field="transaction_datetime",
)

customer_entity = defs.Entity(
    name="customer", description="entity for customers", join_keys=["customer_id"]
)

customer_stats_fv = defs.FeatureView(
    name="customer_stats_fv",
    description="Feature view for customer stats.",
    source=customer_stats_source,
    entity=customer_entity,
    online=False,
    features=[
        defs.Feature(name="amount_sum_7d", dtype=float),
        defs.Feature(name="amount_avg_7d", dtype=float),
        defs.Feature(name="amount_max_7d", dtype=float),
        defs.Feature(name="amount_std_7d", dtype=float),
        defs.Feature(name="unique_merchants_7d", dtype=int),
        defs.Feature(name="unique_categories_7d", dtype=int),
        defs.Feature(name="unique_states_7d", dtype=int),
        defs.Feature(name="unique_cities_7d", dtype=int),
    ],
)

fraud_fs = defs.FeatureService(
    name="fraud_feature_service_v1",
    description="Fraud detection system feature service",
    feature_views=[customer_stats_fv],
    tags={"version": "v1"},
)


if __name__ == "__main__":
    # 1. load in historical training data
    # 2. load in fraud_fs
    # 3. Call fraud_fs.get_historical_features(entity_df)
    # 4. Return data

    from rich import print

    loader = data.ParquetLoader(
        path="data/training/inputs_train.parquet", dataframe_type="pandas"
    )
    entity_df = loader.load()

    training_df = fraud_fs.get_historical_features(entity_df=entity_df)
    print(training_df.shape)
    print(training_df.columns)
    print(training_df)
