from __future__ import annotations

import abc
import contextlib as ctx
import sys
import typing as T

import loguru
import mlflow
import mlflow.tracking as mt
import pydantic as pdt


class Runtime(abc.ABC, pdt.BaseModel, strict=True, frozen=True, extra="forbid"):
    """Base class for all runtime services."""

    @abc.abstractmethod
    def start(self) -> None:
        """Start runtime service."""
        pass

    def stop(self) -> None:
        """Stop runtime service."""
        pass


class Logger(Runtime):
    """Project logger.

    Args:
        sink (str): Sink to log messages. Defaults to "stderr".
        level (str): Level to log messages. Defaults to "DEBUG".
        format (str): Format of the log message. Defaults to:
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green>"
            "<level>{level: <8}</level>"
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan>"
            " <level>{message}</level>"
        serialize (bool): Serialize the log message. Defaults to False.
        colorize (bool): Colorize the log message. Defaults to True.
        backtrace (bool): Show backtrace. Defaults to True.
        diagnose (bool): Diagnose the log message. Defaults to False.
        catch (bool): Catch exceptions. Defaults to True.
    """

    sink: str = "stderr"
    level: str = "DEBUG"
    format: str = (
        "<green>[{time:YYYY-MM-DD HH:mm:ss.SSS}]</green> | "
        "<level>[{level}]</level> | "
        "<cyan>[{name}</cyan>:<magenta>{function}</magenta>:<yellow>{line}]</yellow> | "
        "<level>{message}</level>"
    )
    serialize: bool = False
    colorize: bool = True
    backtrace: bool = True
    diagnose: bool = False
    catch: bool = True

    def start(self) -> None:
        loguru.logger.remove()
        config = self.model_dump()

        sink_mapping = {
            "stderr": sys.stderr,
            "sys.stderr": sys.stderr,
            "stdout": sys.stdout,
            "sys.stdout": sys.stdout,
        }
        config["sink"] = sink_mapping.get(config["sink"], config["sink"])
        loguru.logger.add(**config)

    def stop(self) -> None:
        pass

    def logger(self) -> loguru.Logger:
        """Return the logger.

        Returns:
            loguru.Logger: logger object.
        """
        return loguru.logger


class Mlflow(Runtime):
    """Service for Mlflow tracking and registry.

    Parameters:
        tracking_uri (str): the URI for the Mlflow tracking server.
        registry_uri (str): the URI for the Mlflow model registry.
        experiment_name (str): the name of tracking experiment.
        registry_name (str): the name of model registry.
        autolog_disable (bool): disable autologging.
        autolog_disable_for_unsupported_versions (bool): disable autologging for unsupported versions.
        autolog_exclusive (bool): If True, enables exclusive autologging.
        autolog_log_input_examples (bool): If True, logs input examples during autologging.
        autolog_log_model_signatures (bool): If True, logs model signatures during autologging.
        autolog_log_models (bool): If True, enables logging of models during autologging.
        autolog_log_datasets (bool): If True, logs datasets used during autologging.
        autolog_silent (bool): If True, suppresses all Mlflow warnings during autologging.
    """

    class RunConfig(pdt.BaseModel, strict=True, frozen=True, extra="forbid"):
        """Run configuration for Mlflow tracking.

        Parameters:
            name (str): name of the run.
            description (str | None): description of the run.
            tags (dict[str, T.Any] | None): tags for the run.
            log_system_metrics (bool | None): enable system metrics logging.
        """

        name: str
        description: str | None = None
        tags: dict[str, T.Any] | None = None
        log_system_metrics: bool | None = True

    # server uri
    tracking_uri: str = "http://localhost:5001"
    registry_uri: str = "http://localhost:5001"
    # experiment
    experiment_name: str = "fraudsys"
    # registry
    registry_name: str = "fraudsys"
    # autolog
    autolog_disable: bool = False
    autolog_disable_for_unsupported_versions: bool = False
    autolog_exclusive: bool = False
    autolog_log_input_examples: bool = True
    autolog_log_model_signatures: bool = True
    autolog_log_models: bool = False
    autolog_log_datasets: bool = False
    autolog_silent: bool = False

    @T.override
    def start(self) -> None:
        # server uri
        mlflow.set_tracking_uri(uri=self.tracking_uri)
        mlflow.set_registry_uri(uri=self.registry_uri)
        # experiment
        mlflow.set_experiment(experiment_name=self.experiment_name)
        # autolog
        mlflow.autolog(
            disable=self.autolog_disable,
            disable_for_unsupported_versions=self.autolog_disable_for_unsupported_versions,
            exclusive=self.autolog_exclusive,
            log_input_examples=self.autolog_log_input_examples,
            log_model_signatures=self.autolog_log_model_signatures,
            log_datasets=self.autolog_log_datasets,
            silent=self.autolog_silent,
        )

    @ctx.contextmanager
    def run_context(
        self, run_config: RunConfig
    ) -> T.Generator[mlflow.ActiveRun, None, None]:
        """Yield an active Mlflow run and exit it afterwards.

        Args:
            run (str): run parameters.

        Yields:
            T.Generator[mlflow.ActiveRun, None, None]: active run context. Will be closed at the end of context.
        """
        with mlflow.start_run(
            run_name=run_config.name,
            tags=run_config.tags,
            description=run_config.description,
            log_system_metrics=run_config.log_system_metrics,
        ) as run:
            yield run

    def client(self) -> mt.MlflowClient:
        """Return a new Mlflow client.

        Returns:
            MlflowClient: the mlflow client.
        """
        return mt.MlflowClient(
            tracking_uri=self.tracking_uri, registry_uri=self.registry_uri
        )
