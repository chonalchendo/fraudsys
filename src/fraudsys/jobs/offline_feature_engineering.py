import typing as T
from copy import deepcopy

import polars as pl
import pydantic as pdt

from fraudsys.core import features
from fraudsys.io import datasets
from fraudsys.jobs import base


class OfflineFeatureEngineeringJob(base.DataJob):
    """Batch feature engineering job to pre-compute aggregated features for model
    training and online inference.

    This job creates separate feature views for each time window and feature type,
    which can then be combined in Feast feature services for training and inference.
    """

    KIND: T.Literal["offline_feature_engineering"] = "offline_feature_engineering"

    # Input data sources
    inputs: datasets.LoaderKind = pdt.Field(..., discriminator="KIND")
    targets: datasets.LoaderKind = pdt.Field(..., discriminator="KIND")

    # Output writers for different feature types
    customer_stats_output: datasets.WriterKind = pdt.Field(..., discriminator="KIND")
    merchant_stats_output: datasets.WriterKind = pdt.Field(..., discriminator="KIND")
    customer_behavior_output: datasets.WriterKind = pdt.Field(..., discriminator="KIND")

    # Time windows to compute features for
    windows: dict[str, str] = pdt.Field(
        default={"1h": "1h", "6h": "6h", "1d": "1d", "7d": "7d", "30d": "30d"},
        description="Time windows for feature computation (name -> period)",
    )

    @T.override
    def run(self) -> base.Locals:
        """Run the offline feature engineering job."""
        logger = self.logger.logger()
        logger.info("Starting offline feature engineering job")

        # 1. Load input data
        logger.info("Loading transaction data from: {}", self.inputs.path)
        inputs_df = self.inputs.load()
        logger.info("Loaded transaction data shape: {}", inputs_df.shape)

        logger.info("Loading targets data from: {}", self.targets.path)
        targets_df = self.targets.load()
        logger.info("Loaded targets data shape: {}", targets_df.shape)

        # Convert to Polars if needed
        if hasattr(inputs_df, "reset_index"):
            inputs_df = inputs_df.reset_index()
        if hasattr(targets_df, "reset_index"):
            targets_df = targets_df.reset_index()

        if not isinstance(inputs_df, pl.DataFrame):
            inputs_pl = pl.from_pandas(inputs_df)
        else:
            inputs_pl = inputs_df

        if not isinstance(targets_df, pl.DataFrame):
            targets_pl = pl.from_pandas(targets_df)
        else:
            targets_pl = targets_df

        # 2. Prepare data for feature computation
        logger.info("Preparing data for feature computation")
        prepared_df = self._prepare_data(inputs_pl)
        logger.info("Prepared data shape: {}", prepared_df.shape)

        # 3. Compute features for each feature type and window
        output_paths = {}

        # Customer transaction stats
        logger.info("Computing customer transaction statistics")
        customer_paths = self._compute_and_save_features(
            prepared_df,
            None,
            "customer_stats",
            self.customer_stats_output,
            features.compute_customer_transaction_stats,
        )
        output_paths["customer_stats"] = customer_paths

        # Merchant stats (requires targets for fraud rates)
        logger.info("Computing merchant statistics")
        merchant_paths = self._compute_and_save_features(
            prepared_df,
            targets_pl,
            "merchant_stats",
            self.merchant_stats_output,
            features.compute_merchant_stats,
        )
        output_paths["merchant_stats"] = merchant_paths

        # Customer behavior stats
        logger.info("Computing customer behavior statistics")
        behavior_paths = self._compute_and_save_features(
            prepared_df,
            None,
            "customer_behavior",
            self.customer_behavior_output,
            features.compute_customer_behavior_stats,
        )
        output_paths["customer_behavior"] = behavior_paths

        logger.info("Offline feature engineering completed successfully")

        return {
            "output_paths": output_paths,
            "windows_processed": list(self.windows.keys()),
            "feature_types_created": [
                "customer_stats",
                "merchant_stats",
                "customer_behavior",
            ],
        }

    def _prepare_data(self, df: pl.DataFrame) -> pl.DataFrame:
        """Prepare data for feature computation."""
        logger = self.logger.logger()

        # Convert transaction_time to datetime if it's string
        if df["transaction_time"].dtype == pl.Utf8:
            df = df.with_columns(
                [
                    pl.col("transaction_time")
                    .str.to_datetime("%Y-%m-%d %H:%M:%S")
                    .alias("transaction_datetime")
                ]
            )
        else:
            df = df.with_columns(
                [pl.col("transaction_time").alias("transaction_datetime")]
            )

        # Sort by customer and time for proper temporal ordering
        df = df.sort(["customer_id", "transaction_datetime"])

        logger.debug("Data preparation completed")
        return df

    def _compute_and_save_features(
        self,
        df: pl.DataFrame,
        targets_df: pl.DataFrame | None,
        feature_type: str,
        base_output: datasets.WriterKind,
        feature_function: T.Callable,
    ) -> dict[str, str]:
        """Compute and save features for all windows for a specific feature type."""
        logger = self.logger.logger()
        output_paths = {}

        for window_name, window_period in self.windows.items():
            logger.info(
                "Computing {} features for {} window", feature_type, window_name
            )

            # Call the appropriate feature function
            if targets_df is not None:
                # For merchant stats that need fraud labels
                window_features = feature_function(
                    df, targets_df, window_period, window_name
                )
            else:
                # For customer stats and behavior that don't need labels
                window_features = feature_function(df, window_period, window_name)

            logger.info(
                "Created {} {} features: {} rows",
                window_name,
                feature_type,
                window_features.height,
            )

            # Create writer for this specific window and feature type
            window_writer = self._create_window_writer(
                base_output, feature_type, window_name
            )

            # Save features using the writer
            window_writer.write(window_features)
            output_paths[window_name] = window_writer.path
            logger.info(
                "Saved {} {} features to: {}",
                window_name,
                feature_type,
                window_writer.path,
            )

        return output_paths

    def _create_window_writer(
        self, base_output: datasets.WriterKind, window_name: str
    ) -> datasets.WriterKind:
        """Create a writer for a specific feature type and window."""
        # Deep copy the output writer to avoid modifying the original
        window_writer: datasets.WriterKind = deepcopy(base_output)

        # Modify the path to include feature type and window name
        base_path = window_writer.path

        # Insert feature type and window name before file extension
        if base_path.endswith(".parquet"):
            new_path = base_path.replace(".parquet", f"_{window_name}.parquet")
        else:
            new_path = f"{base_path}_{window_name}.parquet"

        window_writer.path = new_path
        return window_writer
