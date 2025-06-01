import typing as T

import pydantic as pdt

from fraudsys.core import features
from fraudsys.io import datasets
from fraudsys.jobs import base


class OfflineFeaturesJob(base.DataJob):
    KIND: T.Literal["offline_features"] = "offline_features"

    input: datasets.LoaderKind = pdt.Field(..., discriminator="KIND")
    output: datasets.WriterKind = pdt.Field(..., discriminator="KIND")

    @T.override
    def run(self) -> base.Locals:
        logger = self.logger.logger()

        logger.info("Loading raw training data...")
        data = self.input.load()

        logger.info("Cleaning data...")
        cleaned_df = features.clean(data)

        logger.info("Outputting cleaned data for model training...")
        self.output.write(cleaned_df)

        logger.debug("PROCESSED DATA:\n{}", cleaned_df)
        logger.debug("COLUMNS:\n{}", cleaned_df.schema)

        logger.info("[SUCCESS] Job complete.")
        return locals()
