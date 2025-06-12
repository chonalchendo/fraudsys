"""Define a job for explaining the model structure and decisions."""

# %% IMPORTS

import typing as T

import pydantic as pdt

from fraudsys.core import schemas
from fraudsys.io import datasets, registries
from fraudsys.jobs import base

if T.TYPE_CHECKING:
    from fraudsys.core import models

# %% JOBS


class ExplanationsJob(base.ModelJob):
    """Generate explanations from the model and a data sample."""

    KIND: T.Literal["explanation"] = "explanation"

    # Samples
    inputs_samples: datasets.LoaderKind = pdt.Field(..., discriminator="KIND")
    # Explanations
    model_explanations: datasets.WriterKind = pdt.Field(..., discriminator="KIND")
    samples_explanations: datasets.WriterKind = pdt.Field(..., discriminator="KIND")
    # Model
    alias_or_version: str | int = "Champion"
    # Loader
    reader: registries.ReaderKind = pdt.Field(
        registries.CustomReader(), discriminator="KIND"
    )

    @T.override
    def run(self) -> base.Locals:
        # services
        logger = self.logger.logger()
        logger.info("With logger: {}", logger)
        # inputs
        logger.info("Load samples: {}", self.inputs_samples)
        inputs_samples = self.inputs_samples.load()  # unchecked!
        inputs_samples = schemas.InputsSchema.check(inputs_samples)
        logger.debug("- Inputs samples shape: {}", inputs_samples.shape)
        # model
        logger.info("With model: {}", self.mlflow_runtime.registry_name)
        model_uri = registries.uri_for_model_alias_or_version(
            name=self.mlflow_runtime.registry_name,
            alias_or_version=self.alias_or_version,
        )
        logger.debug("- Model URI: {}", model_uri)
        # loader
        logger.info("Load model: {}", self.reader)
        model: models.ModelKind = (
            self.reader.read(uri=model_uri).model.unwrap_python_model().model
        )
        logger.debug("- Model: {}", model)
        # explanations
        # - models
        logger.info("Explain model: {}", model)
        model_explanations = model.explain_model()
        logger.debug("- Models explanations shape: {}", model_explanations.shape)
        # # - samples
        logger.info("Explain samples: {}", len(inputs_samples))
        samples_explanations = model.explain_samples(inputs=inputs_samples)
        logger.debug("- Samples explanations shape: {}", samples_explanations.shape)
        # write
        # - model
        logger.info("Write models explanations: {}", self.model_explanations)
        self.model_explanations.write(data=model_explanations)
        # - samples
        logger.info("Write samples explanations: {}", self.samples_explanations)
        self.samples_explanations.write(data=samples_explanations)
        return locals()
