import typing as T

import mlflow
import pydantic as pdt

from fraudsys import data
from fraudsys.features import validation
from fraudsys.infra.mlflow import client as mlflow_client
from fraudsys.jobs import base
from fraudsys.ml import metrics, models, searchers, splitters

"""
Purpose:
- automate the experimentation of different models on the data
- log results of how different baseline models perform on the data
- select the best models to go forward for hyperparameter tuning
- Only use the train data to do cross-validation on
- Validation data to be used when tuning
- Test data to be used when training the final production model
"""


class ExperimentJob(base.ModelJob):
    KIND: T.Literal["experiment"] = "experiment"

    inputs: data.LoaderKind = pdt.Field(..., discriminator="KIND")
    targets: data.LoaderKind = pdt.Field(..., discriminator="KIND")

    model_selection: list[models.Models]

    # Metric
    metric: metrics.MetricKind = pdt.Field(
        metrics.SklearnMetric(), discriminator="KIND"
    )
    # splitter
    splitter: splitters.SplitterKind = pdt.Field(
        splitters.TimeSeriesSplitter(), discriminator="KIND"
    )
    # Searcher
    searcher: searchers.SearcherKind = pdt.Field(..., discriminator="KIND")

    @T.override
    def run(self) -> base.Locals:
        """Run the tuning job in context."""
        # services
        # - logger
        logger = self.logger.logger()
        # mlflow
        client = self.mlflow_runtime.client()
        logger.info("With logger: {}", logger)

        for model in self.model_selection:
            run_name = f"experiment-{model.KIND.replace('_', '-')}"
            with self.mlflow_runtime.run_context(
                run_config=mlflow_client.Mlflow.RunConfig(name=run_name)
            ) as run:
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
                mlflow.log_input(dataset=inputs_lineage, context=run_name)
                logger.debug("- Inputs lineage: {}", inputs_lineage.to_dict())
                # - targets
                logger.info("Log lineage: targets")
                targets_lineage = self.targets.lineage(
                    data=targets,
                    name="targets",
                    targets=validation.TargetsSchema.is_fraud,
                )
                mlflow.log_input(dataset=targets_lineage, context=run_name)
                logger.debug("- Targets lineage: {}", targets_lineage.to_dict())

                # model
                logger.info("With model: {}", model)
                # metric
                logger.info("With metric: {}", self.metric)
                # splitter
                logger.info("With splitter: {}", self.splitter)
                # searcher
                logger.info("Run searcher: {}", self.searcher)
                results, avg_score, params = self.searcher.search(
                    model=model,
                    metric=self.metric,
                    inputs=inputs,
                    targets=targets,
                    cv=self.splitter,
                )
                logger.debug("- Results: \n{}", results)
                logger.debug("- Average F1 Score: {}", avg_score)
                logger.debug("- Params: {}", params)

                client.log_param(run_id=run.info.run_id, key="model", value=model.KIND)
                client.log_param(
                    run_id=run.info.run_id, key="splitter", value=self.splitter.KIND
                )
                client.log_param(
                    run_id=run.info.run_id, key="metric", value=self.metric.KIND
                )
                client.log_param(
                    run_id=run.info.run_id, key="searcher", value=self.searcher.KIND
                )

                client.log_param(
                    run_id=run.info.run_id,
                    key=f"{model.KIND}_avg_f1_score",
                    value=avg_score,
                )
                client.log_param(
                    run_id=run.info.run_id, key=f"{model.KIND}_params", value=params
                )
        return locals()
