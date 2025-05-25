"""Base for high-level project jobs."""

# %% IMPORTS

import abc
import types as TS
import typing as T

import pydantic as pdt

from fraudsys import logging

# %% TYPES

# Local job variables
Locals = T.Dict[str, T.Any]

# %% JOBS


class Job(abc.ABC, pdt.BaseModel, strict=True, frozen=True, extra="forbid"):
    KIND: str

    @abc.abstractmethod
    def run(self) -> Locals:
        """Run the job in context.

        Returns:
            Locals: local job variables.
        """


class DataJob(Job):
    """Base class for data specific jobs.

    Provides context management for logger and alerts services.
    """

    logger_service: logging.Logger = logging.Logger()
    # alerts_service: services.AlertsService = services.AlertsService()

    def __enter__(self) -> T.Self:
        """Enter the job context.

        Returns:
            T.Self: return the current object.
        """
        self.logger_service.start()
        logger = self.logger_service.logger()
        logger.debug("[START] Logger service: {}", self.logger_service)

        # logger.debug("[START] Alerts service: {}", self.alerts_service)
        # self.alerts_service.start()
        return self

    def __exit__(
        self,
        exc_type: T.Type[BaseException] | None,
        exc_value: BaseException | None,
        exc_traceback: TS.TracebackType | None,
    ) -> T.Literal[False]:
        """Exit the job context.

        Args:
            exc_type (T.Type[BaseException] | None): ignored.
            exc_value (BaseException | None): ignored.
            exc_traceback (TS.TracebackType | None): ignored.

        Returns:
            T.Literal[False]: always propagate exceptions.
        """
        logger = self.logger_service.logger()
        # logger.debug("[STOP] Alerts service: {}", self.alerts_service)
        # self.alerts_service.stop()
        logger.debug("[STOP] Logger service: {}", self.logger_service)
        self.logger_service.stop()
        return False  # re-raise


class ModelJob(Job):
    """Base class for a Modelling job.

    use a job to execute runs in  context.
    e.g., to define common services like logger

    Parameters:
        logger_service (services.LoggerService): manage the logger system.
        alerts_service (services.AlertsService): manage the alerts system.
        mlflow_service (services.MlflowService): manage the mlflow system.
    """

    logger_service: logging.Logger = logging.Logger()
    # alerts_service: services.AlertsService = services.AlertsService()
    # mlflow_service: services.MlflowService = services.MlflowService()

    def __enter__(self) -> T.Self:
        """Enter the job context.

        Returns:
            T.Self: return the current object.
        """
        self.logger_service.start()
        logger = self.logger_service.logger()
        logger.debug("[START] Logger service: {}", self.logger_service)
        # logger.debug("[START] Alerts service: {}", self.alerts_service)
        # self.alerts_service.start()
        # logger.debug("[START] Mlflow service: {}", self.mlflow_service)
        # self.mlflow_service.start()
        return self

    def __exit__(
        self,
        exc_type: T.Type[BaseException] | None,
        exc_value: BaseException | None,
        exc_traceback: TS.TracebackType | None,
    ) -> T.Literal[False]:
        """Exit the job context.

        Args:
            exc_type (T.Type[BaseException] | None): ignored.
            exc_value (BaseException | None): ignored.
            exc_traceback (TS.TracebackType | None): ignored.

        Returns:
            T.Literal[False]: always propagate exceptions.
        """
        logger = self.logger_service.logger()
        # logger.debug("[STOP] Mlflow service: {}", self.mlflow_service)
        # self.mlflow_service.stop()
        # logger.debug("[STOP] Alerts service: {}", self.alerts_service)
        # self.alerts_service.stop()
        logger.debug("[STOP] Logger service: {}", self.logger_service)
        self.logger_service.stop()
        return False  # re-raise
