"""Savers, loaders, and registers for model registries."""

# %% IMPORTS

import abc
import typing as T

import mlflow
import pydantic as pdt
from mlflow.pyfunc import PyFuncModel, PythonModel, PythonModelContext

from fraudsys.features import validation
from fraudsys.infra.mlflow import signers
from fraudsys.ml import models

# %% TYPES

# Results of model registry operations
Info: T.TypeAlias = mlflow.models.model.ModelInfo
Alias: T.TypeAlias = mlflow.entities.model_registry.ModelVersion
Version: T.TypeAlias = mlflow.entities.model_registry.ModelVersion

# %% HELPERS


def uri_for_model_alias(name: str, alias: str) -> str:
    """Create a model URI from a model name and an alias.

    Args:
        name (str): name of the mlflow registered model.
        alias (str): alias of the registered model.

    Returns:
        str: model URI as "models:/name@alias".
    """
    return f"models:/{name}@{alias}"


def uri_for_model_version(name: str, version: int) -> str:
    """Create a model URI from a model name and a version.

    Args:
        name (str): name of the mlflow registered model.
        version (int): version of the registered model.

    Returns:
        str: model URI as "models:/name/version."
    """
    return f"models:/{name}/{version}"


def uri_for_model_alias_or_version(name: str, alias_or_version: str | int) -> str:
    """Create a model URi from a model name and an alias or version.

    Args:
        name (str): name of the mlflow registered model.
        alias_or_version (str | int): alias or version of the registered model.

    Returns:
        str: model URI as "models:/name@alias" or "models:/name/version" based on input.
    """
    if isinstance(alias_or_version, int):
        return uri_for_model_version(name=name, version=alias_or_version)
    return uri_for_model_alias(name=name, alias=alias_or_version)


# %% SAVERS


class Saver(abc.ABC, pdt.BaseModel, strict=True, frozen=True, extra="forbid"):
    """Base class for saving models in registry.

    Separate model definition from serialization.
    e.g., to switch between serialization flavors.

    Parameters:
        path (str): model path inside the Mlflow store.
    """

    KIND: str

    path: str = "model"

    @abc.abstractmethod
    def save(
        self,
        model: models.Model,
        signature: signers.Signature,
        input_example: validation.Inputs,
    ) -> Info:
        """Save a model in the model registry.

        Args:
            model (models.Model): project model to save.
            signature (signers.Signature): model signature.
            input_example (schemas.Inputs): sample of inputs.

        Returns:
            Info: model saving information.
        """


class CustomSaver(Saver):
    """Saver for project models using the Mlflow PyFunc module.

    https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html
    """

    KIND: T.Literal["custom_saver"] = "custom_saver"

    class Adapter(PythonModel):  # type: ignore[misc]
        """Adapt a custom model to the Mlflow PyFunc flavor for saving operations.

        https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html?#mlflow.pyfunc.PythonModel
        """

        def __init__(self, model: models.Model):
            """Initialize the custom saver adapter.

            Args:
                model (models.Model): project model.
            """
            self.model = model

        def predict(
            self,
            context: PythonModelContext,
            model_input: validation.Inputs,
            params: dict[str, T.Any] | None = None,
        ) -> validation.Outputs:
            """Generate predictions with a custom model for the given inputs.

            Args:
                context (mlflow.PythonModelContext): mlflow context.
                model_input (schemas.Inputs): inputs for the mlflow model.
                params (dict[str, T.Any] | None): additional parameters.

            Returns:
                schemas.Outputs: validated outputs of the project model.
            """
            return self.model.predict(inputs=model_input)

    @T.override
    def save(
        self,
        model: models.Model,
        signature: signers.Signature,
        input_example: validation.Inputs,
    ) -> Info:
        adapter = CustomSaver.Adapter(model=model)
        return mlflow.pyfunc.log_model(
            python_model=adapter,
            signature=signature,
            artifact_path=self.path,
            input_example=input_example,
        )


class BuiltinSaver(Saver):
    """Saver for built-in models using an Mlflow flavor module.

    https://mlflow.org/docs/latest/models.html#built-in-model-flavors

    Parameters:
        flavor (str): Mlflow flavor module to use for the serialization.
    """

    KIND: T.Literal["builtin_saver"] = "builtin_saver"

    flavor: str

    @T.override
    def save(
        self,
        model: models.Model,
        signature: signers.Signature,
        input_example: validation.Inputs,
    ) -> Info:
        builtin_model = model.get_internal_model()
        module = getattr(mlflow, self.flavor)
        return module.log_model(
            builtin_model,
            artifact_path=self.path,
            signature=signature,
            input_example=input_example,
        )


SaverKind = CustomSaver | BuiltinSaver

# %% READERS


class Reader(abc.ABC, pdt.BaseModel, strict=True, frozen=True, extra="forbid"):
    """Base class for reading models from registry.

    Separate model definition from deserialization.
    e.g., to switch between deserialization flavors.
    """

    KIND: str

    class Adapter(abc.ABC):
        """Adapt any model for the project inference."""

        @abc.abstractmethod
        def predict(self, inputs: validation.Inputs) -> validation.Outputs:
            """Generate predictions with the internal model for the given inputs.

            Args:
                inputs (schemas.Inputs): validated inputs for the project model.

            Returns:
                schemas.Outputs: validated outputs of the project model.
            """

    @abc.abstractmethod
    def read(self, uri: str) -> "Reader.Adapter":
        """Read a model from the model registry.

        Args:
            uri (str): URI of a model to read.

        Returns:
            Reader.Adapter: model read.
        """


class CustomReader(Reader):
    """Reader for custom models using the Mlflow PyFunc module.

    https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html
    """

    KIND: T.Literal["custom_reader"] = "custom_reader"

    class Adapter(Reader.Adapter):
        """Adapt a custom model for the project inference."""

        def __init__(self, model: PyFuncModel) -> None:
            """Initialize the adapter from an mlflow pyfunc model.

            Args:
                model (PyFuncModel): mlflow pyfunc model.
            """
            self.model = model

        @T.override
        def predict(self, inputs: validation.Inputs) -> validation.Outputs:
            # model validation is already done in predict
            outputs = self.model.predict(data=inputs)
            return T.cast(validation.Outputs, outputs)

    @T.override
    def read(self, uri: str) -> "CustomReader.Adapter":
        model = mlflow.pyfunc.load_model(model_uri=uri)
        adapter = CustomReader.Adapter(model=model)
        return adapter


class BuiltinReader(Reader):
    """Reader for built-in models using the Mlflow PyFunc module.

    Note: use Mlflow PyFunc instead of flavors to use standard API.

    https://mlflow.org/docs/latest/models.html#built-in-model-flavors
    """

    KIND: T.Literal["builtin_reader"] = "builtin_reader"

    class Adapter(Reader.Adapter):
        """Adapt a builtin model for the project inference."""

        def __init__(self, model: PyFuncModel) -> None:
            """Initialize the adapter from an mlflow pyfunc model.

            Args:
                model (PyFuncModel): mlflow pyfunc model.
            """
            self.model = model

        @T.override
        def predict(self, inputs: validation.Inputs) -> validation.Outputs:
            columns = list(validation.OutputsSchema.to_schema().columns)
            outputs = self.model.predict(data=inputs)  # unchecked data!
            return validation.Outputs(outputs, columns=columns, index=inputs.index)

    @T.override
    def read(self, uri: str) -> "BuiltinReader.Adapter":
        model = mlflow.pyfunc.load_model(model_uri=uri)
        adapter = BuiltinReader.Adapter(model=model)
        return adapter


ReaderKind = CustomReader | BuiltinReader

# %% REGISTERS


class Register(abc.ABC, pdt.BaseModel, strict=True, frozen=True, extra="forbid"):
    """Base class for registring models to a location.

    Separate model definition from its registration.
    e.g., to change the model registry backend.

    Parameters:
        tags (dict[str, T.Any]): tags for the model.
    """

    KIND: str

    tags: dict[str, T.Any] = {}

    @abc.abstractmethod
    def register(self, name: str, model_uri: str) -> Version:
        """Register a model given its name and URI.

        Args:
            name (str): name of the model to register.
            model_uri (str): URI of a model to register.

        Returns:
            Version: information about the registered model.
        """


class MlflowRegister(Register):
    """Register for models in the Mlflow Model Registry.

    https://mlflow.org/docs/latest/model-registry.html
    """

    KIND: T.Literal["mlflow_register"] = "mlflow_register"

    @T.override
    def register(self, name: str, model_uri: str) -> Version:
        return mlflow.register_model(name=name, model_uri=model_uri, tags=self.tags)


RegisterKind = MlflowRegister
