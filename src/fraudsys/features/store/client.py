import typing as T
from dataclasses import dataclass
from datetime import datetime, timedelta

import ibis
import pandas as pd
from rich import print

from fraudsys import data


class Functions:
    COUNT = "count"
    SUM = "sum"
    MAX = "max"
    MIN = "min"
    MEAN = "mean"
    STD = "std"


@dataclass
class Entity:
    name: str
    description: str
    join_keys: list[str]


@dataclass
class BatchSource:
    name: str
    description: str
    loader: data.Loader
    timestamp_field: datetime

    def read(self, features: list[str] | None = None) -> pd.DataFrame:
        """Read batch data from source within optional date range."""
        data = self.loader.load()
        if features:
            return data[features]
        return data


@dataclass
class StreamingSource:
    pass


@dataclass
class RealTimeSource:
    pass


@dataclass
class Field:
    name: str
    dtype: type
    description: str | None = None


@dataclass
class Aggregate:
    field: Field
    function: T.Literal["sum", "mean", "count", "std", "min", "max"]
    time_window: timedelta

    @property
    def feature_name(self) -> str:
        return self.field.name

    @property
    def feature_alias(self) -> str:
        """Generate feature name from aggregation spec"""
        window_str = (
            f"{self.time_window.days}d"
            if self.time_window.days > 0
            else f"{int(self.time_window.total_seconds() // 3600)}h"
        )
        name = f"{self.field.name}_{self.function}_{window_str}"
        return name


@dataclass
class FeatureView:
    name: str
    description: str
    source: BatchSource
    entity: Entity
    features: list[Aggregate]
    aggregation_interval: timedelta
    timestamp_field: str

    def create_feature_aggregations(self) -> list[ibis.Table]:
        """
        Create aggregated features for each customer at regular intervals
        """
        # load source data
        source_df = self.source.loader.load()
        source_table = ibis.memtable(source_df)

        max_window = max(feature.time_window for feature in self.features)

        # create aggregation intervals
        intervals = (
            source_table.mutate(
                interval_timestamp=source_table[self.timestamp_field].bucket(
                    hours=self.aggregation_interval
                )
            )
            .select([*self.entity.join_keys, "interval_timestamp"])
            .distinct()
        )

        all_data = intervals.join(
            source_table,
            [
                intervals[self.entity.join_keys[0]]
                == source_table[self.entity.join_keys[0]],
                source_table[self.timestamp_field]
                >= (intervals.interval_timestamp - ibis.interval(days=max_window.days)),
                source_table[self.timestamp_field] <= intervals.interval_timestamp,
            ],
        )

        agg_expressions = {}

        for feature in self.features:
            window_condition = all_data[self.timestamp_field] >= (
                all_data.interval_timestamp
                - ibis.interval(days=feature.time_window.days)
            )

            feature_alias = feature.feature_alias

            calcuation_expr = self._generate_calcuation_expression(
                table=all_data, feature=feature, window_condition=window_condition
            )

            agg_expressions[feature_alias] = calcuation_expr

        # Single group-by
        result = all_data.group_by(
            [*self.entity.join_keys, "interval_timestamp"]
        ).aggregate(**agg_expressions)

        print(result.order_by(["customer_id", "interval_timestamp"]))

        return result

    def _generate_calcuation_expression(
        self, table: ibis.Table, feature: Aggregate, window_condition
    ) -> ibis.Value:
        # Build CASE-based aggregation
        feature_col = feature.field.name
        feature_function = feature.function

        match feature_function:
            case Functions.COUNT:
                return ibis.cases((window_condition, 1), else_=0).sum()
            case Functions.SUM:
                return ibis.cases((window_condition, table[feature_col]), else_=0).sum()
            case Functions.MEAN:
                sum_expr = ibis.cases(
                    (window_condition, table[feature_col]), else_=0
                ).sum()
                count_expr = ibis.cases((window_condition, 1), else_=0).sum()
                return sum_expr / count_expr
            case Functions.STD:
                # For std, we need to handle NULL values properly
                return ibis.cases(
                    (window_condition, table[feature_col]), else_=None
                ).std()
            case Functions.MIN:
                return ibis.cases(
                    (window_condition, table[feature_col]), else_=None
                ).min()
            case Functions.MAX:
                return ibis.cases(
                    (window_condition, table[feature_col]), else_=None
                ).max()
            case _:
                raise ibis.IbisError(
                    f"Unrecognised transformation function: {feature_function}"
                )


@dataclass
class FeatureService:
    name: str
    description: str
    feature_views: list[FeatureView]
    tags: dict[str, T.Any]

    def get_historical_features(self, entity_df: pd.DataFrame) -> pd.DataFrame:
        for fv in self.feature_views:
            fv.create_feature_aggregations()

    # hourly_features_renamed = hourly_features.mutate(
    #     transaction_datetime=hourly_features.interval_timestamp
    # )
