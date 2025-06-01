# %% IMPORTS

import abc
import typing as T

import polars as pl
import pydantic as pdt
import xgboost
from imblearn import pipeline
from sklearn import ensemble, linear_model

from fraudsys import constants
from fraudsys.core import pipelines, schemas

# %% TYPES

# Model params
ParamKey = str
ParamValue = T.Any
Params = dict[ParamKey, ParamValue]

# %% MODELS


class Model(abc.ABC, pdt.BaseModel, strict=True, frozen=False, extra="forbid"):
    """Base class for a project model.

    Use a model to adapt AI/ML frameworks.
    e.g., to swap easily one model with another.
    """

    KIND: str

    random_state: int | None = pdt.Field(default=42)

    # variables
    _numericals: list[str] = pdt.PrivateAttr(default=constants.NUMERICAL_COLUMNS)
    _categoricals: list[str] = pdt.PrivateAttr(default=constants.CATEGORICAL_COLUMNS)

    # private
    _pipeline: pipeline.Pipeline | None = pdt.PrivateAttr(default=None)

    def get_params(self, deep: bool = True) -> Params:
        """Get the model params.

        Args:
            deep (bool, optional): ignored.

        Returns:
            Params: internal model parameters.
        """
        params: Params = {}
        for key, value in self.model_dump().items():
            if not key.startswith("_") and not key.isupper():
                params[key] = value
        return params

    def set_params(self, **params: ParamValue) -> T.Self:
        """Set the model params in place.

        Returns:
            T.Self: instance of the model.
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self

    @abc.abstractmethod
    def fit(self, inputs: schemas.Inputs, targets: schemas.Targets) -> T.Self:
        """Fit the model on the given inputs and targets.

        Args:
            inputs (schemas.Inputs): model training inputs.
            targets (schemas.Targets): model training targets.

        Returns:
            T.Self: instance of the model.
        """

    @abc.abstractmethod
    def predict(self, inputs: schemas.Inputs) -> schemas.Outputs:
        """Generate outputs with the model for the given inputs.

        Args:
            inputs (schemas.Inputs): model prediction inputs.

        Returns:
            schemas.Outputs: model prediction outputs.
        """

    def get_internal_model(self) -> T.Any:
        """Return the internal model in the object.

        Raises:
            NotImplementedError: method not implemented.

        Returns:
            T.Any: any internal model (either empty or fitted).
        """
        raise NotImplementedError()


class LogisticRegressionModel(Model):
    KIND: T.Literal["logistic_regression"] = "logistic_regression"

    @T.override
    def fit(
        self, inputs: schemas.Inputs, targets: schemas.Targets
    ) -> "LogisticRegressionModel":
        classifier = linear_model.LogisticRegression(random_state=self.random_state)
        # pipeline
        self._pipeline = pipelines.create_pipeline(
            model=classifier,
            numerical_columns=self._numericals,
            category_columns=self._categoricals,
            random_state=self.random_state,
        )
        self._pipeline.fit(X=inputs, y=targets[schemas.TargetsSchema.is_fraud])
        return self

    @T.override
    def predict(self, inputs: schemas.Inputs) -> schemas.Outputs:
        model = self.get_internal_model()
        prediction = model.predict(inputs)
        outputs_ = inputs.select("instant").with_columns(
            pl.Series("prediction", prediction)
        )
        outputs = schemas.OutputsSchema.check(data=outputs_)
        return outputs

    @T.override
    def get_internal_model(self) -> pipeline.Pipeline:
        model = self._pipeline
        if model is None:
            raise ValueError("Model is not fitted yet!")
        return model


class RandomForestModel(Model):
    KIND: T.Literal["random_forest"] = "random_forest"

    random_state: int | None = pdt.Field(default=42)

    @T.override
    def fit(
        self, inputs: schemas.Inputs, targets: schemas.Targets
    ) -> "RandomForestModel":
        classifier = ensemble.RandomForestClassifier(random_state=self.random_state)
        # pipeline
        self._pipeline = pipelines.create_pipeline(
            model=classifier,
            numerical_columns=self._numericals,
            category_columns=self._categoricals,
            random_state=self.random_state,
        )
        self._pipeline.fit(X=inputs, y=targets[schemas.TargetsSchema.is_fraud])
        return self

    @T.override
    def predict(self, inputs: schemas.Inputs) -> schemas.Outputs:
        model = self.get_internal_model()
        prediction = model.predict(inputs)
        outputs_ = inputs.select("instant").with_columns(
            pl.Series("prediction", prediction)
        )
        outputs = schemas.OutputsSchema.check(data=outputs_)
        return outputs

    @T.override
    def get_internal_model(self) -> pipeline.Pipeline:
        model = self._pipeline
        if model is None:
            raise ValueError("Model is not fitted yet!")
        return model


class XGBoostModel(Model):
    KIND: T.Literal["xgboost"] = "xgboost"

    @T.override
    def fit(self, inputs: schemas.Inputs, targets: schemas.Targets) -> "XGBoostModel":
        classifier = xgboost.XGBClassifier()
        # pipeline
        self._pipeline = pipelines.create_pipeline(
            model=classifier,
            numerical_columns=self._numericals,
            category_columns=self._categoricals,
            random_state=self.random_state,
        )
        self._pipeline.fit(X=inputs, y=targets[schemas.TargetsSchema.is_fraud])
        return self

    @T.override
    def predict(self, inputs: schemas.Inputs) -> schemas.Outputs:
        model = self.get_internal_model()
        prediction = model.predict(inputs)
        outputs_ = inputs.select("instant").with_columns(
            pl.Series("prediction", prediction)
        )
        outputs = schemas.OutputsSchema.check(data=outputs_)
        return outputs

    @T.override
    def get_internal_model(self) -> pipeline.Pipeline:
        model = self._pipeline
        if model is None:
            raise ValueError("Model is not fitted yet!")
        return model


ModelKind = LogisticRegressionModel | RandomForestModel | XGBoostModel
Models = T.Annotated[ModelKind, pdt.Field(discriminator="KIND")]
