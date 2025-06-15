"""Define a job for evaluating registered models with data."""

# %% IMPORTS

import typing as T

import mlflow
import pandas as pd
import pydantic as pdt

from fraudsys.core import metrics as metrics_
from fraudsys.core import schemas
from fraudsys.io import datasets, registries, runtimes
from fraudsys.jobs import base

# %% JOBS


class EvaluationJob(base.ModelJob):
    KIND: T.Literal["evaluation"] = "evaluation"

    # run
    evaluation_type: T.Literal["training_evaluation", "inference_evaluation"]
    run_config: runtimes.Mlflow.RunConfig = runtimes.Mlflow.RunConfig(name="Evaluation")
    # data
    inputs: datasets.LoaderKind = pdt.Field(..., discriminator="KIND")
    targets: datasets.LoaderKind = pdt.Field(..., discriminator="KIND")
    # model
    model_type: str = pdt.Field("classifier")
    alias_or_version: str | int = pdt.Field("Champion")
    # model reader
    reader: registries.ReaderKind = pdt.Field(
        registries.CustomReader(), discriminator="KIND"
    )
    # metrics
    metrics: metrics_.MetricsKind = pdt.Field([metrics_.SklearnMetric()])
    # evaluators
    evaluators: list[str] = pdt.Field(["default"])
    # thresholds
    thresholds: dict[str, metrics_.Threshold] = {
        "f1_score": metrics_.Threshold(threshold=0.5, greater_is_better=True)
    }

    @T.override
    def run(self) -> base.Locals:
        # runtimes
        # - logger
        logger = self.logger.logger()
        logger.info("With logger: {}", logger)
        # - mlflow
        client = self.mlflow_runtime.client()
        logger.info("With client: {}", client.tracking_uri)

        setattr(self.run_config, "name", self.evaluation_type)

        with self.mlflow_runtime.run_context(run_config=self.run_config) as run:
            logger.info("With run context: {}", run.info)
            # data
            # - inputs
            logger.info("Load inputs: {}", self.inputs)
            inputs_ = self.inputs.load()  # unchecked!
            inputs = schemas.InputsSchema.check(inputs_)
            logger.debug("- Inputs shape: {}", inputs.shape)
            # - targets
            logger.info("Load targets: {}", self.targets)
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
            # model
            logger.info("With model: {}", self.mlflow_runtime.registry_name)
            model_uri = registries.uri_for_model_alias_or_version(
                name=self.mlflow_runtime.registry_name,
                alias_or_version=self.alias_or_version,
            )
            logger.debug("- Model URI: {}", model_uri)
            # loader
            logger.info("Load model: {}", self.reader)
            model = self.reader.read(uri=model_uri)
            logger.debug("- Model: {}", model)
            # outputs
            logger.info("Predict outputs: {}", len(inputs))
            outputs = model.predict(inputs=inputs)  # checked
            logger.debug("- Outputs shape: {}", outputs.shape)
            # dataset
            logger.info("Create dataset: inputs & targets & outputs")
            dataset_ = pd.concat([inputs, targets, outputs], axis="columns")
            dataset = mlflow.data.from_pandas(  # type: ignore[attr-defined]
                df=dataset_,
                name="evaluation",
                targets=schemas.TargetsSchema.is_fraud,
                predictions=schemas.OutputsSchema.prediction,
            )
            logger.debug("- Dataset: {}", dataset.to_dict())
            # metrics
            logger.debug("Convert metrics: {}", self.metrics)
            extra_metrics = [metric.to_mlflow() for metric in self.metrics]
            logger.debug("- Extra metrics: {}", extra_metrics)
            # thresholds
            logger.info("Convert thresholds: {}", self.thresholds)
            validation_thresholds = {
                name: threshold.to_mlflow()
                for name, threshold in self.thresholds.items()
            }
            logger.debug("- Validation thresholds: {}", validation_thresholds)
            # evaluations
            logger.info("Compute evaluations: {}", self.model_type)
            evaluations = mlflow.evaluate(
                data=dataset,
                model_type=self.model_type,
                evaluators=self.evaluators,
                extra_metrics=extra_metrics,
                validation_thresholds=validation_thresholds,
            )
            logger.debug("- Evaluations metrics: {}", evaluations.metrics)
        return locals()
