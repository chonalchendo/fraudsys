# fastapi entrypoint
import typing as T

import uvicorn

from fraudsys.services import base
from fraudsys.services.api import app as api_app
from fraudsys.services.api import models


class APIService(base.Service):
    KIND: T.Literal["api"] = "api"

    execute: str
    host: str
    port: int
    reload: bool

    kafka_servers: list[str]
    input_topic: str

    def start(self) -> None:
        ctx = models.AppContext(
            kafka_servers=self.kafka_servers,
            input_topic=self.input_topic,
        )

        api_app.inject_context(ctx)

        uvicorn.run(self.execute, host=self.host, port=self.port, reload=self.reload)
