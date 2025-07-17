# fastapi entrypoint
import typing as T

import uvicorn

from fraudsys.infra import logging
from fraudsys.services import base
from fraudsys.services.api import dependencies as deps
from fraudsys.services.api import models


class APIService(base.Service):
    KIND: T.Literal["api"] = "api"

    execute: str
    host: str
    port: int
    reload: bool

    kafka_servers: list[str]
    raw_transactions_topic: str
    predictions_topic: str

    mlflow_tracking_uri: str
    mlflow_registry: str
    mlflow_model_alias: str

    logger: logging.Logger = logging.Logger()

    def start(self) -> None:
        ctx = models.AppContext(
            kafka_servers=self.kafka_servers,
            raw_transactions_topic=self.raw_transactions_topic,
            predictions_topic=self.predictions_topic,
            mlflow_tracking_uri=self.mlflow_tracking_uri,
            mlflow_registry=self.mlflow_registry,
            mlflow_model_alias=self.mlflow_model_alias,
            logger=self.logger,
        )

        deps.inject_context(ctx)

        uvicorn.run(self.execute, host=self.host, port=self.port, reload=self.reload)
