import abc
import typing as T
from pathlib import Path

import kaggle
import polars as pl
import pydantic as pdt

# %% - LOADERS

type LoadType = pl.DataFrame | list[pl.DataFrame]
type WriteType = pl.DataFrame


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
            list[dict[str, T.Any]]: The loaded data.
        """
        pass


class JsonLoader(Loader):
    KIND: T.Literal["json"] = "json"

    @T.override
    def load(self) -> pl.DataFrame:
        if not Path(self.path).exists():
            raise FileNotFoundError(f"File not found: {self.path}")
        return pl.read_ndjson(self.path)


class ParquetLoader(Loader):
    KIND: T.Literal["parquet"] = "parquet"

    storage_options: dict | None = pdt.Field(default=None)

    @T.override
    def load(self) -> LoadType:
        if not Path(self.path).exists():
            raise FileNotFoundError(f"File not found: {self.path}")
        return pl.read_parquet(self.path, storage_options=self.storage_options)


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

        return datasets


LoaderKind = JsonLoader | ParquetLoader | KaggleLoader

# %% - WRITERS


class Writer(abc.ABC, pdt.BaseModel, strict=True, frozen=False, extra="forbid"):
    """Base class for all writers.

    Args:
        path (str): Path to write the data.
    """

    KIND: str

    path: str

    @abc.abstractmethod
    def write(self, data: WriteType) -> None:
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

    @T.override
    def write(self, data: WriteType) -> None:
        self._ensure_parent_dir(self.path)
        data.write_parquet(self.path)


WriterKind = ParquetWriter
