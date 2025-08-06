import ibis

from fraudsys.features.store import definitions as defs

if __name__ == "__main__":
    ibis.options.interactive = True
    entity_df = ibis.read_parquet("data/training/inputs_train.parquet")

    training_df = defs.fraud_fs.get_historical_features(entity_df=entity_df)
