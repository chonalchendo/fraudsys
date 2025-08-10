import ibis
from rich import print

from fraudsys.features.store import definitions as defs

if __name__ == "__main__":
    ibis.options.interactive = True
    entity_df = ibis.read_parquet("data/training/inputs_train.parquet")
    entity_df = entity_df.to_pandas()

    training_df = defs.fraud_fs.get_historical_features(entity_df=entity_df)

    print(
        training_df.loc[
            training_df["customer_id"] == "CUST_f198489bae4db7c3",
            [
                "instant",
                "transaction_datetime",
                "amount_usd",
                "amount_usd_count_1d",
                "amount_usd_mean_1d",
                "amount_usd_mean_7d",
                "amount_usd_count_7d",
            ],
        ]
    )
