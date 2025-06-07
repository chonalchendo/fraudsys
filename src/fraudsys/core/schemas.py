"""Define and validate dataframe schemas."""

# %% IMPORTS

import typing as T

import pandas as pd
import pandera.pandas as pa
import pandera.typing as papd
import pandera.typing.common as padt

# %% TYPES

# Generic type for a dataframe container
TSchema = T.TypeVar("TSchema", bound="pa.DataFrameModel")

# %% SCHEMAS


class Schema(pa.DataFrameModel):
    """Base class for a dataframe schema.

    Use a schema to type your dataframe object.
    e.g., to communicate and validate its fields.
    """

    class Config:
        """Default configurations for all schemas.

        Parameters:
            coerce (bool): convert data type if possible.
            strict (bool): ensure the data type is correct.
        """

        coerce: bool = True
        strict: bool = True

    @classmethod
    def check(cls: T.Type[TSchema], data: pd.DataFrame) -> papd.DataFrame[TSchema]:
        """Check the dataframe with this schema.

        Args:
            data (pl.DataFrame): dataframe to check.

        Returns:
            papd.DataFrame[TSchema]: validated dataframe.
        """
        return T.cast(papd.DataFrame[TSchema], cls.validate(data))


class InputsSchema(Schema):
    """Schema for the project inputs."""

    instant: papd.Index[padt.UInt32] = pa.Field(ge=0)
    customer_id: papd.Series[str] = pa.Field()
    transaction_id: papd.Series[str] = pa.Field()
    transaction_time: papd.Series[str] = pa.Field()
    merchant_name: papd.Series[str] = pa.Field()
    category: papd.Series[str] = pa.Field()
    amount_usd: papd.Series[padt.Float32] = pa.Field()
    gender: papd.Series[str] = pa.Field()
    street: papd.Series[str] = pa.Field()
    city: papd.Series[str] = pa.Field()
    state: papd.Series[str] = pa.Field()
    zip: papd.Series[padt.UInt32] = pa.Field()
    lat: papd.Series[padt.Float32] = pa.Field()
    long: papd.Series[padt.Float32] = pa.Field()
    city_pop: papd.Series[padt.UInt32] = pa.Field()
    job: papd.Series[str] = pa.Field()
    dob: papd.Series[str] = pa.Field()
    unix_time: papd.Series[padt.UInt32] = pa.Field()
    merch_lat: papd.Series[padt.Float32] = pa.Field(ge=-90, le=90)
    merch_long: papd.Series[padt.Float32] = pa.Field(ge=-180, le=180)


Inputs = papd.DataFrame[InputsSchema]


class TargetsSchema(Schema):
    """Schema for the project target variable."""

    instant: papd.Index[padt.UInt32] = pa.Field(ge=0)
    is_fraud: papd.Series[padt.UInt8] = pa.Field(isin=[0, 1])


Targets = papd.DataFrame[TargetsSchema]


class OutputsSchema(Schema):
    """Schema for the project output."""

    instant: papd.Index[padt.UInt32] = pa.Field(ge=0)
    prediction: papd.Series[padt.UInt8] = pa.Field(isin=[0, 1])


Outputs = papd.DataFrame[OutputsSchema]
