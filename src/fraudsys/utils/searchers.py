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
Grid = dict[models.ParamKey, T.Sequence[models.ParamValue]]
# Distribution of model params
Dist = dict[models.ParamKey, T.Sequence[models.ParamValue]]

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


class RandomCVSearcher(Searcher):
    KIND: T.Literal["random_cv"] = "random_cv"

    param_dist: Dist

    n_iter: int = pdt.Field(default=10)
    n_jobs: int | None = pdt.Field(default=None)
    refit: bool = pdt.Field(default=True)
    verbose: int = pdt.Field(default=3)
    error_score: str | float = pdt.Field(default="raise")
    return_train_score: bool = pdt.Field(default=False)
    random_state: int | None = pdt.Field(default=42)

    @T.override
    def search(
        self,
        model: models.Model,
        metric: metrics.Metric,
        inputs: schemas.Inputs,
        targets: schemas.Targets,
        cv: CrossValidation,
    ) -> Results:
        searcher = model_selection.RandomizedSearchCV(
            estimator=model,
            n_iter=self.n_iter,
            scoring=metric.scorer,
            cv=cv,
            param_distributions=self.param_dist,
            n_jobs=self.n_jobs,
            refit=self.refit,
            verbose=self.verbose,
            error_score=self.error_score,
            return_train_score=self.return_train_score,
            random_state=self.random_state,
        )
        searcher.fit(inputs, targets)
        results = pl.DataFrame(searcher.cv_results_, strict=False)
        return results, searcher.best_score_, searcher.best_params_


SearcherKind = CVSearcher | RandomCVSearcher
