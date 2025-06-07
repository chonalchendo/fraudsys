import typing as T

import mlflow
import numpy as np
import pydantic as pdt
from rich import print

from fraudsys.core import metrics as metrics_
from fraudsys.core import models, schemas
from fraudsys.io import datasets, registries, runtimes
from fraudsys.jobs import base
from fraudsys.utils import signers, splitters

"""
Questions:
- what data am I pulling into this training pipeline?
    - cleaned data
    - feature engineering occurs within the Model class
- Will I use cross-validation during training so evaluate which models are best?
    - Yes, cross-validation will be included and results logged in MlFlow to
    access which models are best.
- Will the data be split based on datetime and into three splits (train, test, validation)? 
    - Yes, the data will be split into train, test, and validation. The validation set
    will be used to tune hyperparamters in the tune job.
- Will a baseline model be selected with more advanced models being used to see how performance
improves.
    - Yes, a baseline classification model will be used with further models being used to see
    how performance improves but important to find trade off between complexity/simplicity,
    explainability, and inference speed.
- 
"""


class TrainingJob(base.ModelJob):
    KIND: T.Literal["training"] = "training"

    # run
    run_config: runtimes.Mlflow.RunConfig = runtimes.Mlflow.RunConfig(name="Training")

    # data
    inputs: datasets.LoaderKind = pdt.Field(..., discriminator="KIND")
    targets: datasets.LoaderKind = pdt.Field(..., discriminator="KIND")

    # model
    model: models.ModelKind = pdt.Field(..., discriminator="KIND")

    # metrics
    metrics: metrics_.MetricsKind

    # splitter
    splitter: splitters.SplitterKind = pdt.Field(..., discriminator="KIND")

    # saver
    saver: registries.SaverKind = pdt.Field(..., discriminator="KIND")

    # signer
    signer: signers.SignerKind = pdt.Field(..., discriminator="KIND")

    # registry
    registry: registries.RegisterKind = pdt.Field(..., discriminator="KIND")

    @T.override
    def run(self) -> base.Locals:
        # runtimes
        # logger
        logger = self.logger.logger()
        logger.info("With logger: {}", logger)
        # mlflow
        # disable xgboost autologging
        if self.model.KIND == "xgboost":
            mlflow.xgboost.autolog(disable=True)

        client = self.mlflow_runtime.client()
        logger.info("With client: {}", client.tracking_uri)

        with self.mlflow_runtime.run_context(run_config=self.run_config) as run:
            logger.info("With run context: {}", run.info)
            # data
            # - inputs
            logger.info("Load inputs: {}", self.inputs)
            inputs_ = self.inputs.load()  # unchecked!
            inputs = schemas.InputsSchema.check(inputs_)
            logger.debug("- Inputs shape: {}", inputs.shape)
            # - targets
            logger.info("load targets: {}", self.targets)
            targets_ = self.targets.load()  # unchecked!
            targets = schemas.TargetsSchema.check(targets_)
            logger.debug("- Targets shape: {}", targets.shape)
            # lineage
            # - inputs
            logger.info("Log lineage: inputs")
            inputs_lineage = self.inputs.lineage(data=inputs, name="inputs")
            mlflow.log_input(dataset=inputs_lineage, context=self.run_config.name)
            logger.debug("- Inputs lineage: {}", inputs_lineage.to_dict())
            # - targets
            logger.info("Log lineage: targets")
            targets_lineage = self.targets.lineage(
                data=targets, name="targets", targets=schemas.TargetsSchema.is_fraud
            )
            mlflow.log_input(dataset=targets_lineage, context=self.run_config.name)
            logger.debug("- Targets lineage: {}", targets_lineage.to_dict())
            # splitter
            logger.info("With splitter: {}", self.splitter)
            # - index
            train_index, test_index = next(
                self.splitter.split(inputs=inputs, targets=targets)
            )
            # - inputs
            inputs_train = T.cast(schemas.Inputs, inputs.iloc[train_index])
            inputs_test = T.cast(schemas.Inputs, inputs.iloc[test_index])
            logger.debug("- Inputs train shape: {}", inputs_train.shape)
            logger.debug("- Inputs test shape: {}", inputs_test.shape)
            # - targets
            targets_train = T.cast(schemas.Targets, targets.iloc[train_index])
            targets_test = T.cast(schemas.Targets, targets.iloc[test_index])
            logger.debug("- Targets train shape: {}", targets_train.shape)
            logger.debug("- Targets test shape: {}", targets_test.shape)
            # model
            logger.info("Fit model: {}", self.model)
            self.model.fit(inputs=inputs_train, targets=targets_train)
            # outputs
            logger.info("Predict outputs: {}", len(inputs_test))
            outputs_test = self.model.predict(inputs=inputs_test)
            logger.debug("- Outputs test shape: {}", outputs_test.shape)
            # metrics
            for i, metric in enumerate(self.metrics, start=1):
                logger.info("{}. Compute metric: {}", i, metric)
                score = metric.score(targets=targets_test, outputs=outputs_test)
                client.log_metric(run_id=run.info.run_id, key=metric.name, value=score)
                logger.debug("- Metric score: {}", score)
            # signer
            logger.info("Sign model: {}", self.signer)
            model_signature = self.signer.sign(inputs=inputs, outputs=outputs_test)
            logger.debug("- Model signature: {}", model_signature.to_dict())
            # saver
            logger.info("Save model: {}", self.saver)
            model_info = self.saver.save(
                model=self.model, signature=model_signature, input_example=inputs
            )
            logger.debug("- Model URI: {}", model_info.model_uri)
            # register
            logger.info("Register model: {}", self.registry)
            model_version = self.registry.register(
                name=self.mlflow_runtime.registry_name, model_uri=model_info.model_uri
            )
            logger.debug("- Model version: {}", model_version)

        return locals()
