"""Feature engineering job for computing aggregated features for Feast feature store."""

import math
import typing as T
from datetime import datetime, timedelta

import polars as pl
import pydantic as pdt

from fraudsys import data
from fraudsys.jobs import base


class FeatureEngineeringJob(base.DataJob):
    """Feature engineering job for computing aggregated features from base transaction data.

    This job computes customer and merchant aggregated features that are stored
    in the offline feature store and later materialized to the online store for
    real-time inference.
    """

    KIND: T.Literal["feature_engineering"] = "feature_engineering"

    # Input data sources
    inputs: data.LoaderKind = pdt.Field(..., discriminator="KIND")
    targets: data.LoaderKind = pdt.Field(..., discriminator="KIND")

    # Output writers
    customer_stats_writer: data.WriterKind = pdt.Field(..., discriminator="KIND")
    merchant_stats_writer: data.WriterKind = pdt.Field(..., discriminator="KIND")
    customer_behavior_writer: data.WriterKind = pdt.Field(..., discriminator="KIND")

    # Time configuration
    reference_time: str | None = pdt.Field(
        default=None,
        description="Reference time for feature computation (defaults to current time)",
    )

    # Feature computation configuration
    time_windows: dict[str, int] = pdt.Field(
        default={
            "1h": 1,
            "6h": 6,
            "1d": 24,
            "7d": 24 * 7,
            "30d": 24 * 30,
        },
        description="Time windows for aggregations (window_name -> hours)",
    )

    @T.override
    def run(self) -> base.Locals:
        """Run the feature engineering job."""
        logger = self.logger.logger()
        logger.info("Starting feature engineering job")

        # Set reference time if not provided, convert string to datetime if needed
        if self.reference_time is None:
            reference_time = datetime.now()
        elif isinstance(self.reference_time, str):
            reference_time = datetime.fromisoformat(
                self.reference_time.replace("Z", "+00:00")
            )
        else:
            reference_time = self.reference_time
        logger.info("Using reference time: {}", reference_time)

        # Load input data
        logger.info("Loading inputs: {}", self.inputs)
        inputs_df = self.inputs.load()
        logger.debug("Inputs shape: {}", inputs_df.shape)

        logger.info("Loading targets: {}", self.targets)
        targets_df = self.targets.load()
        logger.debug("Targets shape: {}", targets_df.shape)

        # Convert to Polars for efficient computation
        logger.info("Converting to Polars DataFrames")
        # Reset index to preserve it as a column when converting to polars
        if hasattr(inputs_df, "reset_index"):
            inputs_df = inputs_df.reset_index()
        if hasattr(targets_df, "reset_index"):
            targets_df = targets_df.reset_index()

        inputs_pl = (
            pl.from_pandas(inputs_df)
            if hasattr(inputs_df, "to_pandas")
            else pl.from_pandas(inputs_df)
        )
        targets_pl = (
            pl.from_pandas(targets_df)
            if hasattr(targets_df, "to_pandas")
            else pl.from_pandas(targets_df)
        )

        # Compute customer transaction statistics
        logger.info("Computing customer transaction statistics")
        customer_stats = self._compute_customer_transaction_stats(
            inputs_pl, reference_time
        )
        logger.info("Saving customer stats to: {}", self.customer_stats_writer.path)
        self.customer_stats_writer.write(customer_stats)
        logger.debug("Customer stats shape: {}", customer_stats.shape)

        # Compute merchant statistics
        logger.info("Computing merchant statistics")
        merchant_stats = self._compute_merchant_stats(
            inputs_pl, targets_pl, reference_time
        )
        logger.info("Saving merchant stats to: {}", self.merchant_stats_writer.path)
        self.merchant_stats_writer.write(merchant_stats)
        logger.debug("Merchant stats shape: {}", merchant_stats.shape)

        # Compute customer behavior patterns
        logger.info("Computing customer behavior patterns")
        customer_behavior = self._compute_customer_behavior(inputs_pl, reference_time)
        logger.info(
            "Saving customer behavior to: {}", self.customer_behavior_writer.path
        )
        self.customer_behavior_writer.write(customer_behavior)
        logger.debug("Customer behavior shape: {}", customer_behavior.shape)

        logger.info("Feature engineering job completed successfully")

        return {
            "customer_stats": customer_stats,
            "merchant_stats": merchant_stats,
            "customer_behavior": customer_behavior,
            "reference_time": reference_time,
            "output_paths": {
                "customer_stats": self.customer_stats_writer.path,
                "merchant_stats": self.merchant_stats_writer.path,
                "customer_behavior": self.customer_behavior_writer.path,
            },
        }

    def _compute_customer_transaction_stats(
        self, transactions_df: pl.DataFrame, reference_time: datetime
    ) -> pl.DataFrame:
        """Compute customer transaction statistics for different time windows."""
        logger = self.logger.logger()

        # Convert transaction_time to datetime if it's string
        df = transactions_df.with_columns(
            [
                pl.col("transaction_time")
                .str.to_datetime("%Y-%m-%d %H:%M:%S")
                .alias("transaction_datetime")
            ]
        )

        results = []

        for window_name, hours in self.time_windows.items():
            logger.debug("Computing {} window stats", window_name)
            window_start = reference_time - timedelta(hours=hours)

            # Filter to time window
            window_df = df.filter(pl.col("transaction_datetime") >= window_start)

            # Compute aggregations
            stats = window_df.group_by("customer_id").agg(
                [
                    # Transaction counts
                    pl.count().alias(f"transaction_count_{window_name}"),
                    # Amount statistics
                    pl.col("amount_usd").sum().alias(f"total_amount_{window_name}"),
                    pl.col("amount_usd").mean().alias(f"avg_amount_{window_name}"),
                    pl.col("amount_usd").max().alias(f"max_amount_{window_name}"),
                    pl.col("amount_usd").std().alias(f"std_amount_{window_name}"),
                    # Merchant and category diversity
                    pl.col("merchant_name")
                    .n_unique()
                    .alias(f"unique_merchants_{window_name}"),
                    pl.col("category")
                    .n_unique()
                    .alias(f"unique_categories_{window_name}"),
                    # Geographic diversity
                    pl.col("state").n_unique().alias(f"unique_states_{window_name}"),
                    pl.col("city").n_unique().alias(f"unique_cities_{window_name}"),
                ]
            )

            results.append(stats)

        # Join all time window stats
        final_df: pl.DataFrame = results[0]
        for stats_df in results[1:]:
            final_df = final_df.join(
                stats_df, on="customer_id", how="full", coalesce=True
            )

        # Add timing pattern features for 7d window
        df_7d = df.filter(
            pl.col("transaction_datetime") >= (reference_time - timedelta(days=7))
        )

        timing_stats = (
            df_7d.with_columns(
                [
                    pl.col("transaction_datetime").dt.hour().alias("hour"),
                    pl.col("transaction_datetime").dt.weekday().alias("weekday"),
                ]
            )
            .group_by("customer_id")
            .agg(
                [
                    # Night transactions (10PM-6AM)
                    pl.when((pl.col("hour") >= 22) | (pl.col("hour") <= 6))
                    .then(1)
                    .otherwise(0)
                    .sum()
                    .alias("night_transactions_7d"),
                    # Weekend transactions (Saturday=6, Sunday=7 in polars)
                    pl.when(pl.col("weekday").is_in([6, 7]))
                    .then(1)
                    .otherwise(0)
                    .sum()
                    .alias("weekend_transactions_7d"),
                    # Business hours (9AM-5PM)
                    pl.when((pl.col("hour") >= 9) & (pl.col("hour") <= 17))
                    .then(1)
                    .otherwise(0)
                    .sum()
                    .alias("business_hours_transactions_7d"),
                ]
            )
        )

        # Join timing stats
        final_df = final_df.join(timing_stats, on="customer_id", how="left")

        # Add event_timestamp and created_timestamp for Feast
        final_df = final_df.with_columns(
            [
                pl.lit(reference_time).alias("event_timestamp"),
                pl.lit(datetime.now()).alias("created_timestamp"),
            ]
        )

        return final_df

    def _compute_merchant_stats(
        self,
        transactions_df: pl.DataFrame,
        targets_df: pl.DataFrame,
        reference_time: datetime,
    ) -> pl.DataFrame:
        """Compute merchant risk statistics."""
        logger = self.logger.logger()

        # Join transactions with fraud labels
        df = transactions_df.join(
            targets_df.select(["instant", "is_fraud"]), on="instant", how="left"
        )

        # Convert transaction_time to datetime
        df = df.with_columns(
            [
                pl.col("transaction_time")
                .str.to_datetime("%Y-%m-%d %H:%M:%S")
                .alias("transaction_datetime")
            ]
        )

        time_windows = {"1d": 1, "7d": 7, "30d": 30}
        results = []

        for window_name, days in time_windows.items():
            logger.debug("Computing merchant {} window stats", window_name)
            window_start = reference_time - timedelta(days=days)

            window_df = df.filter(pl.col("transaction_datetime") >= window_start)

            stats = window_df.group_by("merchant_name").agg(
                [
                    # Transaction volume
                    pl.count().alias(f"transaction_count_{window_name}"),
                    # Fraud indicators
                    pl.col("is_fraud").mean().alias(f"fraud_rate_{window_name}"),
                    pl.col("is_fraud").sum().alias(f"fraud_count_{window_name}"),
                    # Amount patterns
                    pl.col("amount_usd")
                    .mean()
                    .alias(f"avg_transaction_amount_{window_name}"),
                    pl.col("amount_usd")
                    .max()
                    .alias(f"max_transaction_amount_{window_name}"),
                    pl.col("amount_usd")
                    .std()
                    .alias(f"std_transaction_amount_{window_name}"),
                    # Customer diversity
                    pl.col("customer_id")
                    .n_unique()
                    .alias(f"unique_customers_{window_name}"),
                ]
            )

            results.append(stats)

        # Join all windows
        final_df: pl.DataFrame = results[0]
        for stats_df in results[1:]:
            final_df = final_df.join(
                stats_df, on="merchant_name", how="full", coalesce=True
            )

        # Compute additional risk indicators for 7d window
        df_7d = df.filter(
            pl.col("transaction_datetime") >= (reference_time - timedelta(days=7))
        )

        # High amount transactions (> 95th percentile)
        amount_95th = df_7d.select(pl.col("amount_usd").quantile(0.95)).item()

        high_amount_counts = df_7d.group_by("merchant_name").agg(
            [
                pl.when(pl.col("amount_usd") > amount_95th)
                .then(1)
                .otherwise(0)
                .sum()
                .alias("high_amount_transactions_7d"),
            ]
        )

        final_df = final_df.join(high_amount_counts, on="merchant_name", how="left")

        # Compute customer repeat rates
        customer_stats = (
            df_7d.group_by(["merchant_name", "customer_id"])
            .agg([pl.count().alias("customer_transaction_count")])
            .group_by("merchant_name")
            .agg(
                [
                    pl.when(pl.col("customer_transaction_count") > 1)
                    .then(1)
                    .otherwise(0)
                    .mean()
                    .alias("repeat_customer_rate_7d"),
                    pl.when(pl.col("customer_transaction_count") == 1)
                    .then(1)
                    .otherwise(0)
                    .mean()
                    .alias("new_customer_rate_7d"),
                ]
            )
        )

        final_df = final_df.join(customer_stats, on="merchant_name", how="left")

        # Add timestamps
        final_df = final_df.with_columns(
            [
                pl.lit(reference_time).alias("event_timestamp"),
                pl.lit(datetime.now()).alias("created_timestamp"),
            ]
        )

        return final_df

    def _compute_customer_behavior(
        self,
        transactions_df: pl.DataFrame,
        reference_time: datetime,
    ) -> pl.DataFrame:
        """Compute customer behavioral patterns."""
        logger = self.logger.logger()

        # Convert to datetime and add calculated fields
        df = transactions_df.with_columns(
            [
                pl.col("transaction_time")
                .str.to_datetime("%Y-%m-%d %H:%M:%S")
                .alias("transaction_datetime")
            ]
        )

        # Filter to 7d and 90d windows
        df_7d = df.filter(
            pl.col("transaction_datetime") >= (reference_time - timedelta(days=7))
        )
        df_90d = df.filter(
            pl.col("transaction_datetime") >= (reference_time - timedelta(days=90))
        )

        # Compute velocity features
        velocity_stats = (
            df_7d.sort(["customer_id", "transaction_datetime"])
            .with_columns(
                [
                    pl.col("transaction_datetime")
                    .diff()
                    .dt.total_minutes()
                    .over("customer_id")
                    .alias("time_diff_minutes")
                ]
            )
            .group_by("customer_id")
            .agg(
                [
                    pl.col("time_diff_minutes")
                    .mean()
                    .alias("avg_time_between_transactions_7d"),
                    pl.col("time_diff_minutes")
                    .min()
                    .alias("min_time_between_transactions_7d"),
                    (pl.count() / (7 * 24)).alias(
                        "transaction_velocity_1h"
                    ),  # transactions per hour
                ]
            )
        )

        # Compute location-based features (simplified)
        location_stats = (
            df_7d.group_by("customer_id")
            .agg(
                [
                    # Use customer's location variability as proxy for distance patterns
                    pl.col("lat").std().alias("lat_std"),
                    pl.col("long").std().alias("long_std"),
                ]
            )
            .with_columns(
                [
                    # Approximate distance metrics (simplified calculation)
                    (pl.col("lat_std") * 111).alias(
                        "avg_distance_from_home_7d"
                    ),  # rough km conversion
                    (pl.col("lat_std") * 111 * 1.5).alias("max_distance_from_home_7d"),
                    # Average merchant distance (simplified)
                    (pl.col("lat_std") + pl.col("long_std")).alias(
                        "avg_merchant_distance_7d"
                    ),
                ]
            )
        )

        # Compute spending pattern features
        spending_stats = (
            df_7d.group_by("customer_id")
            .agg(
                [
                    pl.col("amount_usd").mean().alias("avg_amount_7d"),
                    pl.col("amount_usd").std().alias("std_amount_7d"),
                ]
            )
            .with_columns(
                [
                    # Amount deviation (z-score normalized to 0 if std is 0)
                    pl.when(pl.col("std_amount_7d") > 0)
                    .then(
                        0.0
                    )  # Current amount would need to be passed in for real z-score
                    .otherwise(0.0)
                    .alias("amount_deviation_from_avg_7d"),
                    # Amount percentile (simplified to 0.5 for now)
                    pl.lit(0.5).alias("amount_percentile_vs_history"),
                ]
            )
        )

        # Compute category concentration (entropy measure)
        category_stats = (
            df_7d.group_by(["customer_id", "category"])
            .agg([pl.count().alias("category_count")])
            .group_by("customer_id")
            .agg(
                [
                    # Shannon entropy approximation
                    pl.col("category_count")
                    .map_elements(
                        lambda counts: -sum(
                            (c / sum(counts)) * math.log(c / sum(counts))
                            if c > 0
                            else 0
                            for c in counts
                        ),
                        return_dtype=pl.Float64,
                    )
                    .alias("category_concentration_7d"),
                ]
            )
        )

        # Compute recency features
        recency_stats = (
            df_90d.group_by("customer_id")
            .agg(
                [
                    pl.col("transaction_datetime").max().alias("last_transaction"),
                    pl.col("transaction_datetime").min().alias("first_transaction"),
                ]
            )
            .with_columns(
                [
                    (pl.lit(reference_time) - pl.col("last_transaction"))
                    .dt.total_days()
                    .alias("days_since_last_transaction"),
                    (pl.lit(reference_time) - pl.col("first_transaction"))
                    .dt.total_days()
                    .alias("days_since_first_transaction"),
                    (pl.col("last_transaction") - pl.col("first_transaction"))
                    .dt.total_days()
                    .alias("account_age_days"),
                ]
            )
        )

        # Join all stats
        final_df = velocity_stats
        for stats_df in [location_stats, spending_stats, category_stats, recency_stats]:
            final_df = final_df.join(
                stats_df, on="customer_id", how="full", coalesce=True
            )

        # Add timestamps
        final_df = final_df.with_columns(
            [
                pl.lit(reference_time).alias("event_timestamp"),
                pl.lit(datetime.now()).alias("created_timestamp"),
            ]
        )

        return final_df
