import typing as T

import pydantic as pdt

from fraudsys import data
from fraudsys.features.engineering import cleaning
from fraudsys.jobs import base


class OfflineFeaturesJob(base.DataJob):
    KIND: T.Literal["offline_features"] = "offline_features"

    input_raw_train: data.LoaderKind = pdt.Field(..., discriminator="KIND")
    input_raw_test: data.LoaderKind = pdt.Field(..., discriminator="KIND")
    output_inputs_train: data.WriterKind = pdt.Field(..., discriminator="KIND")
    output_inputs_test: data.WriterKind = pdt.Field(..., discriminator="KIND")

    @T.override
    def run(self) -> base.Locals:
        logger = self.logger.logger()

        logger.info("Loading raw training data...")
        raw_train = self.input_raw_train.load()
        raw_test = self.input_raw_test.load()

        logger.info("Cleaning data...")
        cleaned_train = cleaning.clean(raw_train)
        cleaned_test = cleaning.clean(raw_test)

        logger.info("Outputting cleaned data for model training...")
        self.output_inputs_train.write(cleaned_train)
        self.output_inputs_test.write(cleaned_test)

        logger.debug("PROCESSED TRAIN DATA:\n{}", cleaned_train)
        logger.debug("COLUMNS:\n{}", cleaned_train.schema)

        logger.debug("PROCESSED TEST DATA:\n{}", cleaned_test)
        logger.debug("COLUMNS:\n{}", cleaned_test.schema)

        logger.info("[SUCCESS] Job complete.")
        return locals()
