import typing as T

import polars as pl

from fraudsys import logging as logger_
from fraudsys.io import kafka
from fraudsys.services import base


class CleanerService(base.Service):
    KIND: T.Literal["cleaner"] = "cleaner"

    kafka_producer: kafka.KafkaProducerWrapper
    kafka_consumer: kafka.KafkaConsumerWrapper

    rename_columns: dict[str, T.Any]
    drop_columns: list[str]
    categorical_columns: list[str]

    logger: logger_.Logger = logger_.Logger()

    @T.override
    def start(self) -> None:
        self.logger.start()
        logger = self.logger.logger()

        for message in self.kafka_consumer:
            cleaned_message = self._process_message(message.value)
            self.kafka_producer.send(message=cleaned_message)
            logger.debug("CLEANED MESSAGE: {}", cleaned_message)

        self.kafka_producer.get_producer.flush()

    def _process_message(self, message: dict) -> pl.DataFrame:
        data = pl.from_records([message])
        transformed_data = self._clean(data)
        return transformed_data.to_dicts()

    def _clean(self, data: pl.DataFrame) -> pl.DataFrame:
        df = data.__copy__()
        df = df.rename(self.rename_columns).drop(self.drop_columns)
        data = df.with_columns(*self._convert_to_category())
        return data

    def _convert_to_category(self) -> T.Generator[pl.Expr, T.Any, None]:
        columns = self.categorical_columns

        for column in columns:
            yield pl.col(column).cast(pl.Categorical)
