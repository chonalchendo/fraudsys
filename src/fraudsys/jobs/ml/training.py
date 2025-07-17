import typing as T

import mlflow
import pydantic as pdt

from fraudsys import data
from fraudsys.features import validation
from fraudsys.infra.mlflow import client, registries, signers
from fraudsys.jobs import base
from fraudsys.ml import metrics as metrics_
from fraudsys.ml import models, splitters


class TrainingJob(base.ModelJob):
    KIND: T.Literal["training"] = "training"

    # run
    run_config: client.Mlflow.RunConfig = client.Mlflow.RunConfig(name="Training")
    # data
    inputs: data.LoaderKind = pdt.Field(..., discriminator="KIND")
    targets: data.LoaderKind = pdt.Field(..., discriminator="KIND")
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
            inputs = validation.InputsSchema.check(inputs_)
            logger.debug("- Inputs shape: {}", inputs.shape)
            # - targets
            logger.info("load targets: {}", self.targets)
            targets_ = self.targets.load()  # unchecked!
            targets = validation.TargetsSchema.check(targets_)
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
                data=targets, name="targets", targets=validation.TargetsSchema.is_fraud
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
            inputs_train = T.cast(validation.Inputs, inputs.iloc[train_index])
            inputs_test = T.cast(validation.Inputs, inputs.iloc[test_index])
            logger.debug("- Inputs train shape: {}", inputs_train.shape)
            logger.debug("- Inputs test shape: {}", inputs_test.shape)
            # - targets
            targets_train = T.cast(validation.Targets, targets.iloc[train_index])
            targets_test = T.cast(validation.Targets, targets.iloc[test_index])
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
