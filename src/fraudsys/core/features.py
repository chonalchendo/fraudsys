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
