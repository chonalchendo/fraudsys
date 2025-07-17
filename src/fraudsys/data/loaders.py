import abc
import typing as T
from pathlib import Path

import kaggle
import mlflow.data.pandas_dataset as lineage
import pandas as pd
import polars as pl
import pydantic as pdt

# %% - LOADERS

type DataFrameType = pl.DataFrame | pd.DataFrame
type LoadType = pl.DataFrame | pd.DataFrame | tuple[pd.DataFrame, ...]
type WriteType = pd.DataFrame
Lineage: T.TypeAlias = lineage.PandasDataset


class Loader(abc.ABC, pdt.BaseModel, strict=True, frozen=False, extra="forbid"):
    """Base class for all loaders.

    Args:
        path (str): Path to load the data.
    """

    KIND: str

    path: str

    @abc.abstractmethod
    def load(self) -> LoadType:
        """Load the data from the path.

        Returns:
            LoadType: The loaded data.
        """
        pass

    def lineage(
        self,
        name: str,
        data: pl.DataFrame,
        targets: str | None = None,
        predictions: str | None = None,
    ) -> Lineage:
        """Generate lineage information.

        Args:
            name (str): dataset name.
            data (pd.DataFrame): reader dataframe.
            targets (str | None): name of the target column.
            predictions (str | None): name of the prediction column.

        Returns:
            Lineage: lineage information.
        """


class JsonLoader(Loader):
    KIND: T.Literal["json"] = "json"

    @T.override
    def load(self) -> LoadType:
        if not Path(self.path).exists():
            raise FileNotFoundError(f"File not found: {self.path}")
        return pd.read_json(self.path, lines=True)


class ParquetLoader(Loader):
    KIND: T.Literal["parquet"] = "parquet"

    dataframe_type: T.Literal["pandas", "polars"] = pdt.Field(default="polars")
    storage_options: dict | None = pdt.Field(default=None)
    backend: T.Literal["pyarrow", "numpy_nullable"] = pdt.Field(default="pyarrow")
    index_name: str = pdt.Field(default="instant")
    limit: int | None = pdt.Field(default=None)

    @T.override
    def load(self) -> LoadType:
        # Skip existence check for S3 paths as Path().exists() doesn't work with S3
        if not self.path.startswith("s3://") and not Path(self.path).exists():
            raise FileNotFoundError(f"File not found: {self.path}")

        df = pl.read_parquet(
            self.path, storage_options=self.storage_options, n_rows=self.limit
        )

        if self.dataframe_type == "pandas":
            df = df.to_pandas()

            # deal with instant index assignment
            drop_columns = []
            if "" in df.columns:
                drop_columns.append("")
            if "instant" in df.columns:
                drop_columns.append("instant")
            if drop_columns:
                df = df.drop(columns=drop_columns)

            if df.index.name != self.index_name:
                df = df.rename_axis(self.index_name)

            if self.limit is not None:
                df = df.head(self.limit)
            return df

        if self.dataframe_type == "polars":
            if len(df.columns) == 1 and df.columns[0] != self.index_name:
                return df.with_row_index(name=self.index_name)

            if self.index_name not in df.columns:
                return df.rename({df.columns[0]: self.index_name})
            return df

    @T.override
    def lineage(
        self,
        name: str,
        data: DataFrameType,
        targets: str | None = None,
        predictions: str | None = None,
    ) -> Lineage:
        if isinstance(data, pl.DataFrame):
            data = data.to_pandas()
        return lineage.from_pandas(
            df=data,
            name=name,
            source=self.path,
            targets=targets,
            predictions=predictions,
        )


class KaggleLoader(Loader):
    KIND: T.Literal["kaggle"] = "kaggle"

    dataset_path: str
    output_directory: str
    unzip: bool = pdt.Field(default=True)

    @T.override
    def load(self) -> LoadType:
        kaggle.api.authenticate()

        Path(self.output_directory).mkdir(exist_ok=True, parents=True)

        kaggle.api.dataset_download_files(
            dataset=self.dataset_path, path=self.output_directory, unzip=self.unzip
        )

        paths = Path(self.output_directory).glob("*.csv")

        datasets = []
        for path in paths:
            if not Path(path).exists():
                raise FileNotFoundError(f"Data did not download correctly: {path}")

            data = pl.read_csv(path)

            # save file as parquet
            output = path.with_suffix(".parquet")
            data.write_parquet(str(output))

            # delete original csv files
            path.unlink()

            datasets.append(data)

        return tuple(datasets)


LoaderKind = JsonLoader | ParquetLoader | KaggleLoader
