"""Define a job for generating batch predictions from a registered model."""

# %% IMPORTS

import typing as T

import pydantic as pdt

from fraudsys.core import schemas
from fraudsys.io import datasets, registries
from fraudsys.jobs import base

# %% JOBS


class InferenceJob(base.ModelJob):
    """Generate predictions from a registered model on test set.

    The predictions from this job will then be used to evaluate
    if the model is good enough to be used in production.

    Parameters:
        inputs (datasets.ReaderKind): reader for the inputs data.
        outputs (datasets.WriterKind): writer for the outputs data.
        alias_or_version (str | int): alias or version for the  model.
        loader (registries.LoaderKind): registry loader for the model.
    """

    KIND: T.Literal["inference"] = "inference"

    # Inputs
    inputs: datasets.LoaderKind = pdt.Field(..., discriminator="KIND")
    # Outputs
    outputs: datasets.WriterKind = pdt.Field(..., discriminator="KIND")
    # Model
    alias_or_version: str | int = "Champion"
    # Reader
    reader: registries.ReaderKind = pdt.Field(
        registries.CustomReader(), discriminator="KIND"
    )

    @T.override
    def run(self) -> base.Locals:
        # services
        logger = self.logger.logger()
        logger.info("With logger: {}", logger)
        # inputs
        logger.info("Load inputs: {}", self.inputs)
        inputs_ = self.inputs.load()  # unchecked!
        inputs = schemas.InputsSchema.check(inputs_)
        logger.debug("- Inputs shape: {}", inputs.shape)
        # model
        logger.info("With model: {}", self.mlflow_runtime.registry_name)
        model_uri = registries.uri_for_model_alias_or_version(
            name=self.mlflow_runtime.registry_name,
            alias_or_version=self.alias_or_version,
        )
        logger.debug("- Model URI: {}", model_uri)
        # reader
        logger.info("Load model: {}", self.reader)
        model = self.reader.read(uri=model_uri)
        logger.debug("- Model: {}", model)
        # outputs
        logger.info("Predict outputs: {}", len(inputs))
        outputs = model.predict(inputs=inputs)  # checked
        logger.debug("- Outputs shape: {}", outputs.shape)
        # write
        logger.info("Write outputs: {}", self.outputs)
        self.outputs.write(data=outputs)
        return locals()
