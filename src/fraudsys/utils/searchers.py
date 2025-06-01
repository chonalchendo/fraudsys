"""Find the best hyperparameters for a model."""

# %% IMPORTS

import abc
import typing as T

import polars as pl
import pydantic as pdt
from sklearn import model_selection

from fraudsys.core import metrics, models, schemas
from fraudsys.utils import splitters

# %% TYPES

# Grid of model params
Grid = dict[models.ParamKey, list[models.ParamValue]]

# Results of a model search
Results = tuple[
    T.Annotated[pl.DataFrame, "details"],
    T.Annotated[float, "best score"],
    T.Annotated[models.Params, "params"],
]

# Cross-validation options for searchers
CrossValidation = int | splitters.TrainTestSplits | splitters.Splitter

# %% SEARCHERS


class Searcher(abc.ABC, pdt.BaseModel, strict=True, frozen=True, extra="forbid"):
    """Base class for a searcher.

    Use searcher to fine-tune models.
    i.e., to find the best model params.

    Parameters:
        param_grid (Grid): mapping of param key -> values.
    """

    KIND: str

    @abc.abstractmethod
    def search(
        self,
        model: models.Model,
        metric: metrics.Metric,
        inputs: schemas.Inputs,
        targets: schemas.Targets,
        cv: CrossValidation,
    ) -> Results:
        """Search the best model for the given inputs and targets.

        Args:
            model (models.Model): AI/ML model to fine-tune.
            metric (metrics.Metric): main metric to optimize.
            inputs (schemas.Inputs): model inputs for tuning.
            targets (schemas.Targets): model targets for tuning.
            cv (CrossValidation): choice for cross-fold validation.

        Returns:
            Results: all the results of the searcher execution process.
        """


class CVSearcher(Searcher):
    KIND: T.Literal["cross_validation"] = "cross_validation"

    @T.override
    def search(
        self,
        model: models.Model,
        metric: metrics.Metric,
        inputs: schemas.Inputs,
        targets: schemas.Targets,
        cv: CrossValidation,
    ) -> Results:
        scores = model_selection.cross_val_score(
            estimator=model, X=inputs, y=targets, scoring=metric.scorer, cv=cv
        )
        params = model.get_params()
        results = pl.DataFrame({"cv_scores": scores})
        results.sort(by="cv_scores", descending=True)
        avg_score = results["cv_scores"].mean()
        return results, avg_score, params


SearcherKind = CVSearcher
