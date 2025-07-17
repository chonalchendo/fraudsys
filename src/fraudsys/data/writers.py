import abc
import typing as T
from pathlib import Path

import pandas as pd
import polars as pl
import pydantic as pdt

# %% - LOADERS

type DataFrameType = pl.DataFrame | pd.DataFrame
type WriteType = pd.DataFrame


# %% - WRITERS


class Writer(abc.ABC, pdt.BaseModel, strict=True, frozen=False, extra="forbid"):
    """Base class for all writers.

    Args:
        path (str): Path to write the data.
    """

    KIND: str

    path: str

    @abc.abstractmethod
    def write(self, data: pd.DataFrame) -> None:
        """Write the data to the path.

        Args:
            data (WriteType): Data to write.
        """
        pass

    def _ensure_parent_dir(self, output_path: str) -> None:
        """Ensure parent directory exists."""
        if not Path(output_path).parent.absolute().exists():
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)


class ParquetWriter(Writer):
    KIND: T.Literal["parquet"] = "parquet"

    dataframe_type: T.Literal["pandas", "polars"] = pdt.Field("polars")
    storage_options: dict | None = pdt.Field(default=None)

    @T.override
    def write(self, data: DataFrameType) -> None:
        if "s3://" not in self.path:
            self._ensure_parent_dir(self.path)

        if self.dataframe_type == "pandas":
            data.to_parquet(
                self.path, index=False, storage_options=self.storage_options
            )

        if self.dataframe_type == "polars":
            data.write_parquet(self.path, storage_options=self.storage_options)


WriterKind = ParquetWriter
