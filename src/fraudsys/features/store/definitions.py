# where logic is defined - like Dagster definitions fil
import typing as T
from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from fraudsys import data


@dataclass
class BatchSource:
    name: str
    description: str
    loader: data.Loader
    timestamp_field: datetime

    def read(self, features: list[str]) -> pd.DataFrame:
        """Read batch data from source within optional date range."""
        data = self.loader.load()
        return data[features]


@dataclass
class Entity:
    name: str
    description: str
    join_keys: list[str]


@dataclass
class Feature:
    name: str
    dtype: type
    description: str | None = None


@dataclass
class FeatureView:
    name: str
    description: str
    source: BatchSource
    entity: Entity
    features: list[Feature]
    online: bool

    @property
    def get_dtypes(self) -> dict[str, type]:
        return {feat.name: feat.dtype for feat in self.features}

    @property
    def get_feature_names(self) -> list[str]:
        """Get list of all feature names in this view."""
        return (
            [self.get_timestamp_field]
            + self.entity.join_keys
            + [feat.name for feat in self.features]
            + ["event_timestamp"]
        )

    @property
    def get_timestamp_field(self) -> str:
        return self.source.timestamp_field

    @property
    def get_entity_join_keys(self) -> list[str]:
        return self.entity.join_keys


@dataclass
class FeatureService:
    name: str
    description: str
    feature_views: list[FeatureView]
    tags: dict[str, T.Any]

    def get_historical_features(self, entity_df: pd.DataFrame) -> pd.DataFrame:
        """Get historical features from all views, joined on entity keys."""
        entity_df_ = entity_df.copy()

        for view in self.feature_views:
            feats = view.get_feature_names
            data = view.source.read(feats)

            entity_df_ = pd.merge_asof(
                left=entity_df_.sort_values(by=view.get_timestamp_field),
                right=data.sort_values(by=view.get_timestamp_field),
                by=view.get_entity_join_keys,
                on=view.get_timestamp_field,
                direction="backward",
            )

        return entity_df_
