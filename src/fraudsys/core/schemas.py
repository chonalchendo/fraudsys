"""Define and validate dataframe schemas."""

# %% IMPORTS

import typing as T

import pandera.polars as pa
import pandera.typing.common as padt
import pandera.typing.polars as papl
import polars as pl

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
    def check(cls: T.Type[TSchema], data: pl.DataFrame) -> papl.DataFrame[TSchema]:
        """Check the dataframe with this schema.

        Args:
            data (pl.DataFrame): dataframe to check.

        Returns:
            papd.DataFrame[TSchema]: validated dataframe.
        """
        return T.cast(papl.DataFrame[TSchema], cls.validate(data))


class InputsSchema(Schema):
    instant: papl.Series[padt.UInt32] = pa.Field(ge=0)
    customer_id: papl.Series[str] = pa.Field()
    transaction_id: papl.Series[str] = pa.Field()
    transaction_time: papl.Series[str] = pa.Field()
    merchant_name: papl.Series[str] = pa.Field()
    category: papl.Series[str] = pa.Field()
    amount_usd: papl.Series[padt.Float32] = pa.Field()
    gender: papl.Series[str] = pa.Field()
    street: papl.Series[str] = pa.Field()
    city: papl.Series[str] = pa.Field()
    state: papl.Series[str] = pa.Field()
    zip: papl.Series[padt.UInt64] = pa.Field()
    lat: papl.Series[padt.Float32] = pa.Field()
    long: papl.Series[padt.Float32] = pa.Field()
    city_pop: papl.Series[padt.UInt32] = pa.Field()
    job: papl.Series[str] = pa.Field()
    dob: papl.Series[str] = pa.Field()
    unix_time: papl.Series[padt.UInt32] = pa.Field()
    merch_lat: papl.Series[padt.Float32] = pa.Field(ge=-90, le=90)
    merch_long: papl.Series[padt.Float32] = pa.Field(ge=-180, le=180)


Inputs = papl.DataFrame[InputsSchema]


class TargetsSchema(Schema):
    instant: papl.Series[padt.UInt32] = pa.Field(ge=0)
    is_fraud: papl.Series[padt.Bool] = pa.Field()


Targets = papl.DataFrame[TargetsSchema]


class OutputsSchema(Schema):
    """Schema for the project output."""

    instant: papl.Series[padt.UInt32] = pa.Field(ge=0)
    prediction: papl.Series[padt.Bool] = pa.Field(isin=[True, False])


Outputs = papl.DataFrame[OutputsSchema]
