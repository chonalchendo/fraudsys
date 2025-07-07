import typing as T

import polars as pl

from fraudsys import constants


def clean(data: pl.DataFrame) -> pl.DataFrame:
    df = data.__copy__()
    df = df.rename(constants.RENAME_COLUMNS).drop(constants.DROP_COLUMNS)
    data = df.with_columns(*_convert_to_category())
    return data


def _convert_to_category() -> T.Generator[pl.Expr, T.Any, None]:
    for column in constants.CATEGORICAL_COLUMNS:
        yield pl.col(column).cast(pl.Categorical)


# ============================================================================
# Batch Feature Aggregation Functions
# ============================================================================


def compute_customer_transaction_stats(
    df: pl.DataFrame, window_period: str, window_name: str
) -> pl.DataFrame:
    """Compute customer transaction statistics for a specific time window.

    Args:
        df: Transaction data with transaction_datetime and customer_id columns
        window_period: Time period for grouping (e.g., "1h", "7d")
        window_name: Name for feature suffixes (e.g., "1h", "7d")

    Returns:
        DataFrame with customer transaction statistics per time window
    """
    features = df.group_by_dynamic(
        "transaction_datetime",
        group_by="customer_id",
        every=window_period,
        closed="left",  # Don't include current period to avoid data leakage
    ).agg(
        [
            # Transaction count and frequency
            pl.len().alias(f"transaction_count_{window_name}"),
            # Amount statistics
            pl.col("amount_usd").sum().alias(f"amount_sum_{window_name}"),
            pl.col("amount_usd").mean().alias(f"amount_avg_{window_name}"),
            pl.col("amount_usd").std().alias(f"amount_std_{window_name}"),
            pl.col("amount_usd").max().alias(f"amount_max_{window_name}"),
            pl.col("amount_usd").min().alias(f"amount_min_{window_name}"),
            pl.col("amount_usd").median().alias(f"amount_median_{window_name}"),
            # Diversity metrics
            pl.col("merchant_name").n_unique().alias(f"unique_merchants_{window_name}"),
            pl.col("category").n_unique().alias(f"unique_categories_{window_name}"),
            pl.col("state").n_unique().alias(f"unique_states_{window_name}"),
            pl.col("city").n_unique().alias(f"unique_cities_{window_name}"),
            # Geographic patterns
            pl.col("lat").std().alias(f"lat_std_{window_name}"),
            pl.col("long").std().alias(f"long_std_{window_name}"),
            # Temporal patterns
            pl.when(
                pl.col("transaction_datetime")
                .dt.hour()
                .is_between(22, 6, closed="both")
            )
            .then(1)
            .otherwise(0)
            .sum()
            .alias(f"night_transactions_{window_name}"),
            pl.when(pl.col("transaction_datetime").dt.weekday().is_in([6, 7]))
            .then(1)
            .otherwise(0)
            .sum()
            .alias(f"weekend_transactions_{window_name}"),
        ]
    )

    # Add derived features
    features = features.with_columns(
        [
            # Transaction velocity (transactions per day)
            (
                pl.col(f"transaction_count_{window_name}")
                / _get_window_days(window_period)
            ).alias(f"transaction_velocity_{window_name}"),
            # Geographic diversity (approximate distance metric)
            (pl.col(f"lat_std_{window_name}") * 111)  # rough km conversion
            .alias(f"geographic_spread_{window_name}"),
            # Amount concentration (coefficient of variation)
            (
                pl.col(f"amount_std_{window_name}")
                / pl.col(f"amount_avg_{window_name}")
            ).alias(f"amount_cv_{window_name}"),
        ]
    )

    # Add event_timestamp for Feast compatibility
    features = features.with_columns(
        [pl.col("transaction_datetime").alias("event_timestamp")]
    )

    return features


def compute_merchant_stats(
    df: pl.DataFrame, targets_df: pl.DataFrame, window_period: str, window_name: str
) -> pl.DataFrame:
    """Compute merchant risk statistics for a specific time window.

    Args:
        df: Transaction data
        targets_df: Fraud labels data
        window_period: Time period for grouping (e.g., "1d", "7d")
        window_name: Name for feature suffixes

    Returns:
        DataFrame with merchant risk statistics per time window
    """
    # Join transactions with fraud labels
    df_with_labels = df.join(
        targets_df.select(["instant", "is_fraud"]), on="instant", how="left"
    )

    # Sort by merchant and time for group_by_dynamic
    df_with_labels = df_with_labels.sort(["merchant_name", "transaction_datetime"])

    features = df_with_labels.group_by_dynamic(
        "transaction_datetime",
        group_by="merchant_name",
        every=window_period,
        closed="left",
    ).agg(
        [
            # Transaction volume
            pl.len().alias(f"transaction_count_{window_name}"),
            # Fraud indicators
            pl.col("is_fraud").mean().alias(f"fraud_rate_{window_name}"),
            pl.col("is_fraud").sum().alias(f"fraud_count_{window_name}"),
            # Amount patterns
            pl.col("amount_usd").mean().alias(f"avg_transaction_amount_{window_name}"),
            pl.col("amount_usd").max().alias(f"max_transaction_amount_{window_name}"),
            pl.col("amount_usd").std().alias(f"std_transaction_amount_{window_name}"),
            pl.col("amount_usd").sum().alias(f"total_amount_{window_name}"),
            # Customer diversity
            pl.col("customer_id").n_unique().alias(f"unique_customers_{window_name}"),
            # Geographic patterns
            pl.col("state").n_unique().alias(f"unique_states_{window_name}"),
            pl.col("city").n_unique().alias(f"unique_cities_{window_name}"),
        ]
    )

    # Add derived features
    features = features.with_columns(
        [
            # Risk indicators
            (
                pl.col(f"fraud_count_{window_name}")
                / pl.col(f"transaction_count_{window_name}")
            ).alias(f"fraud_density_{window_name}"),
            # Customer concentration
            (
                pl.col(f"transaction_count_{window_name}")
                / pl.col(f"unique_customers_{window_name}")
            ).alias(f"avg_transactions_per_customer_{window_name}"),
            # Amount velocity
            (
                pl.col(f"total_amount_{window_name}") / _get_window_days(window_period)
            ).alias(f"daily_amount_velocity_{window_name}"),
        ]
    )

    # Add event_timestamp for Feast compatibility
    features = features.with_columns(
        [pl.col("transaction_datetime").alias("event_timestamp")]
    )

    return features


def compute_customer_behavior_stats(
    df: pl.DataFrame, window_period: str, window_name: str
) -> pl.DataFrame:
    """Compute customer behavioral patterns for a specific time window.

    Args:
        df: Transaction data
        window_period: Time period for grouping (e.g., "7d", "30d")
        window_name: Name for feature suffixes

    Returns:
        DataFrame with customer behavioral statistics per time window
    """
    # Ensure data is sorted by customer and time for group_by_dynamic
    df = df.sort(["customer_id", "transaction_datetime"])

    features = df.group_by_dynamic(
        "transaction_datetime",
        group_by="customer_id",
        every=window_period,
        closed="left",
    ).agg(
        [
            # Basic transaction patterns
            pl.len().alias(f"transaction_count_{window_name}"),
            pl.col("amount_usd").mean().alias(f"avg_amount_{window_name}"),
            pl.col("amount_usd").std().alias(f"std_amount_{window_name}"),
            # Velocity patterns (time between transactions)
            pl.col("transaction_datetime")
            .diff()
            .dt.total_minutes()
            .mean()
            .alias(f"avg_time_between_transactions_{window_name}"),
            pl.col("transaction_datetime")
            .diff()
            .dt.total_minutes()
            .min()
            .alias(f"min_time_between_transactions_{window_name}"),
            # Location patterns
            pl.col("lat").std().alias(f"location_lat_std_{window_name}"),
            pl.col("long").std().alias(f"location_long_std_{window_name}"),
            # Merchant patterns
            pl.col("merchant_name").n_unique().alias(f"unique_merchants_{window_name}"),
            pl.col("category").n_unique().alias(f"unique_categories_{window_name}"),
            # Timing patterns
            pl.col("transaction_datetime")
            .dt.hour()
            .std()
            .alias(f"hour_std_{window_name}"),
            pl.col("transaction_datetime")
            .dt.weekday()
            .std()
            .alias(f"weekday_std_{window_name}"),
        ]
    )

    # Add derived behavioral features
    features = features.with_columns(
        [
            # Location mobility (approximate)
            (pl.col(f"location_lat_std_{window_name}") * 111).alias(
                f"location_mobility_km_{window_name}"
            ),
            # Transaction frequency
            (
                pl.col(f"transaction_count_{window_name}")
                / _get_window_days(window_period)
            ).alias(f"daily_transaction_frequency_{window_name}"),
            # Spending consistency
            (
                pl.col(f"std_amount_{window_name}")
                / pl.col(f"avg_amount_{window_name}")
            ).alias(f"spending_consistency_{window_name}"),
            # Behavioral diversity score (combination of location and merchant diversity)
            (
                pl.col(f"unique_merchants_{window_name}")
                + pl.col(f"unique_categories_{window_name}")
            ).alias(f"behavioral_diversity_{window_name}"),
        ]
    )

    # Add event_timestamp for Feast compatibility
    features = features.with_columns(
        [pl.col("transaction_datetime").alias("event_timestamp")]
    )

    return features


def _get_window_days(window_period: str) -> float:
    """Convert window period to approximate days for calculations."""
    if "h" in window_period:
        hours = int(window_period.replace("h", ""))
        return hours / 24.0
    elif "d" in window_period:
        return float(window_period.replace("d", ""))
    else:
        # Default fallback
        return 1.0
