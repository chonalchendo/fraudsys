import time
import typing as T

import httpx
import polars as pl
import pydantic as pdt

from fraudsys import data
from fraudsys.infra import logging
from fraudsys.services import base


class ProducerService(base.Service):
    KIND: T.Literal["producer"] = "producer"

    health_check_endpoint: str
    endpoint: str

    input: data.LoaderKind = pdt.Field(..., discriminator="KIND")

    logger: logging.Logger = logging.Logger()

    @T.override
    def start(self) -> None:
        self.logger.start()
        logger = self.logger.logger()

        data = self.input.load()

        if not isinstance(data, pl.DataFrame):
            raise ValueError("Data must be a polars DataFrame")

        self._wait_for_api()

        for record in data.to_dicts():
            try:
                response = httpx.post(self.endpoint, json=record)
                response.raise_for_status()
                logger.debug("POSTED: {}", record)
                logger.debug("RESPONSE: {}", response)
                time.sleep(1)
            except httpx.HTTPStatusError as e:
                logger.error("Network error: {}", e)
                time.sleep(1)

    def _wait_for_api(self, retries: int = 30, delay: int = 2) -> None:
        logger = self.logger.logger()
        for attempt in range(retries):
            try:
                response = httpx.get(self.health_check_endpoint)
                if response.status_code == 200:
                    logger.success(f"API is ready after {attempt + 1} tries.")
                    return
                else:
                    logger.error(
                        f"API healthcheck returned {response.status_code}, retrying..."
                    )
            except httpx.ConnectError:
                print("API not reachable, retrying...")
            time.sleep(delay)
        raise RuntimeError("API not ready after maximum retries.")
