"""Define a job for finding the best hyperparameters for a model."""

# %% IMPORTS

import typing as T

import mlflow
import pydantic as pdt

from fraudsys import data
from fraudsys.features import validation
from fraudsys.infra.mlflow import client
from fraudsys.jobs import base
from fraudsys.ml import metrics, models, searchers, splitters

# %% JOBS


class TuningJob(base.ModelJob):
    KIND: T.Literal["tuning"] = "tuning"

    # run
    run_config: client.Mlflow.RunConfig = client.Mlflow.RunConfig(name="Tuning")

    # data
    inputs: data.LoaderKind = pdt.Field(..., discriminator="KIND")
    targets: data.LoaderKind = pdt.Field(..., discriminator="KIND")

    # model
    model: models.ModelKind = pdt.Field(..., discriminator="KIND")

    # metric
    metric: metrics.MetricKind = pdt.Field(
        metrics.SklearnMetric(), discriminator="KIND"
    )

    # splitter
    splitter: splitters.SplitterKind = pdt.Field(
        splitters.TimeSeriesSplitter(), discriminator="KIND"
    )

    # searcher
    searcher: searchers.SearcherKind = pdt.Field(..., discriminator="KIND")

    @T.override
    def run(self) -> base.Locals:
        """Run the tuning job in context"""
        # services
        # - logger
        logger = self.logger.logger()
        logger.info("With logger: {}", logger)
        with self.mlflow_runtime.run_context(run_config=self.run_config) as run:
            logger.info("With run context: {}", run.info)
            # data
            # - inputs
            logger.info("Load inputs: {}", self.inputs)
            inputs_ = self.inputs.load()  # unchecked!
            inputs = validation.InputsSchema.check(inputs_)
            logger.debug("- Inputs shape: {}", inputs.shape)
            # - targets
            logger.info("Load targets: {}", self.targets)
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
            # model
            logger.info("With model: {}", self.model)
            # metric
            logger.info("With metric: {}", self.metric)
            # splitter
            logger.info("With splitter: {}", self.splitter)
            # searcher
            logger.info("Run searcher: {}", self.searcher)
            results, best_score, best_params = self.searcher.search(
                model=self.model,
                metric=self.metric,
                inputs=inputs,
                targets=targets,
                cv=self.splitter,
            )
            logger.debug("- Results: {}", results)
            logger.debug("- Best Score: {}", best_score)
            logger.debug("- Best Params: {}", best_params)
        return locals()
