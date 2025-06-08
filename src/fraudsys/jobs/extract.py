import typing as T
from hashlib import sha256

import polars as pl
import pydantic as pdt

from fraudsys import constants
from fraudsys.io import datasets
from fraudsys.jobs import base


class ExtractJob(base.DataJob):
    KIND: T.Literal["extract"] = "extract"

    input: datasets.LoaderKind = pdt.Field(..., discriminator="KIND")

    output_inputs_train: datasets.WriterKind = pdt.Field(..., discriminator="KIND")
    output_targets_train: datasets.WriterKind = pdt.Field(..., discriminator="KIND")
    output_inputs_test: datasets.WriterKind = pdt.Field(..., discriminator="KIND")
    output_targets_test: datasets.WriterKind = pdt.Field(..., discriminator="KIND")
    output_inputs_production: datasets.WriterKind = pdt.Field(..., discriminator="KIND")
    output_targets_production: datasets.WriterKind = pdt.Field(
        ..., discriminator="KIND"
    )

    @T.override
    def run(self) -> base.Locals:
        logger = self.logger.logger()

        logger.info("Loading data from Kaggle...")
        data = self.input.load()

        if isinstance(data, pl.DataFrame):
            raise RuntimeError("Expected two datasets (train/test), got one.")

        if len(data) != 2:
            raise RuntimeError(f"Expected two datasets. Got {len(data)}")

        train, test = data

        logger.info("Merging train and test datasets...")
        merged_df = self._merge_datasets(train, test)

        logger.info("Generating customer IDs...")
        merged_dff = self._generate_ids(merged_df)

        logger.info("Splitting datasets...")
        offline_df, online_df = self._split_datasets(merged_dff)

        logger.info("Splitting training data into train and test...")
        train_df, test_df = self._split_training_data(offline_df)

        logger.info("Separating labels from training data...")
        inputs_train, targets_train = self._get_labels(train_df)

        logger.info("Separating labels from testing data...")
        inputs_test, targets_test = self._get_labels(test_df)

        logger.info("Separating labels from production data...")
        inputs_prod, targets_prod = self._get_labels(online_df)

        logger.info("Writing out offline training data...")
        self.output_inputs_train.write(inputs_train)
        self.output_targets_train.write(targets_train)

        logger.info("Writing out offline testing data...")
        self.output_inputs_test.write(inputs_test)
        self.output_targets_test.write(targets_test)

        logger.info("Writing out production data...")
        self.output_inputs_production.write(inputs_prod)
        self.output_targets_production.write(targets_prod)

        return locals()

    def _generate_ids(self, merged_df: pl.DataFrame):
        def _create_customer_id(row: dict) -> str:
            identifier = (
                f"{str(row['first']).lower().strip()}"
                f"{str(row['last']).lower().strip()}"
                f"{str(row['cc_num']).strip()}"
                f"{str(row['dob'])}"
            )
            # Create deterministic hash
            return f"CUST_{sha256(identifier.encode()).hexdigest()[:16]}"

        return merged_df.with_columns(
            pl.struct(constants.SENSITIVE_COLUMNS)
            .map_elements(lambda x: _create_customer_id(x), return_dtype=pl.String)
            .alias("customer_id")
        )

    def _split_datasets(
        self, merged_df: pl.DataFrame
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        train_df = merged_df.filter(pl.col("original_split") == "train").drop(
            "original_split"
        )
        test_df = merged_df.filter(pl.col("original_split") == "test").drop(
            "original_split"
        )
        return train_df, test_df

    def _merge_datasets(
        self, train_df: pl.DataFrame, test_df: pl.DataFrame
    ) -> pl.DataFrame:
        return pl.concat(
            [
                train_df.with_columns(pl.lit("train").alias("original_split")),
                test_df.with_columns(pl.lit("test").alias("original_split")),
            ]
        )

    def _get_labels(self, prod_df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
        is_fraud_df = prod_df.select(constants.TARGET_COLUMN)
        features_df = prod_df.drop(constants.TARGET_COLUMN)
        return features_df, is_fraud_df

    def _split_training_data(
        self, offline_df: pl.DataFrame
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        df = offline_df.sort("trans_date_trans_time")

        # Compute index cutoff
        cutoff_idx = int(0.8 * df.height)

        train_df = df[:cutoff_idx]
        test_df = df[cutoff_idx:]
        return train_df, test_df
