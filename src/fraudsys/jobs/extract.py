import typing as T
from hashlib import sha256

import polars as pl
import pydantic as pdt

from fraudsys.io import datasets
from fraudsys.jobs import base


class ExtractJob(base.DataJob):
    KIND: T.Literal["extract"] = "extract"

    input: datasets.LoaderKind = pdt.Field(..., discriminator="KIND")

    output_training_df: datasets.WriterKind = pdt.Field(..., discriminator="KIND")
    output_production_df: datasets.WriterKind = pdt.Field(..., discriminator="KIND")
    output_production_labels_df: datasets.WriterKind = pdt.Field(
        ..., discriminator="KIND"
    )

    sensitive_columns: list[str]
    target_column: str

    @T.override
    def run(self) -> base.Locals:
        logger = self.logger_service.logger()

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
        train_df, test_df = self._split_datasets(merged_dff)

        logger.info("Separating labels from production data...")
        prod_df, labels_df = self._get_labels(test_df)

        logger.info("Writing out training data...")
        self.output_training_df.write(train_df)

        logger.info("Writing out production data...")
        self.output_production_df.write(prod_df)
        self.output_production_labels_df.write(labels_df)

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
            pl.struct(self.sensitive_columns)
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
        is_fraud_df = prod_df.select(self.target_column)
        features_df = prod_df.drop(self.target_column)
        return features_df, is_fraud_df
