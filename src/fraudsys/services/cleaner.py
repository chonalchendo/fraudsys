import json
import typing as T
from time import sleep

import polars as pl
from kafka import KafkaConsumer, KafkaProducer

from fraudsys import logging as logger_
from fraudsys.services import base


class CleanerService(base.Service):
    KIND: T.Literal["cleaner"] = "cleaner"

    input_topic: str
    output_topic: str
    servers: list[str]
    group_id: str

    rename_columns: dict[str, T.Any]
    drop_columns: list[str]
    categorical_columns: list[str]

    logger: logger_.Logger = logger_.Logger()

    @T.override
    def start(self) -> None:
        self.logger.start()
        logger = self.logger.logger()

        consumer = self._init_consumer()
        producer = self._init_producer()

        for message in consumer:
            cleaned_message = self._process_message(message.value)
            self._send_message(
                producer=producer, topic=self.output_topic, message=cleaned_message
            )
            logger.debug("CLEANED MESSAGE: {}", cleaned_message)

        producer.flush()

    def _process_message(self, message: dict) -> pl.DataFrame:
        data = pl.from_records([message])
        transformed_data = self._clean(data)
        return transformed_data.to_dicts()

    def _send_message(
        self, producer: KafkaProducer, topic: str, message: pl.DataFrame
    ) -> None:
        producer.send(topic, value=message)

    def _init_consumer(self) -> KafkaConsumer:
        logger = self.logger.logger()
        for i in range(20):
            try:
                consumer = KafkaConsumer(
                    self.input_topic,
                    bootstrap_servers=self.servers,
                    group_id=self.group_id,
                    value_deserializer=lambda v: json.loads(v.decode("utf-8")),
                )
                logger.success(
                    "SUCCESS: Initiated Kafka conusmer for topic: {}", self.input_topic
                )
                return consumer
            except Exception as e:
                logger.exception(
                    f"Trying to instantiate Kafka consumer with bootstrap servers {self.servers} with error {e}"
                )
                sleep(10)
                pass

    def _init_producer(self) -> KafkaProducer:
        return KafkaProducer(
            bootstrap_servers=self.servers,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        )

    def _clean(self, data: pl.DataFrame) -> pl.DataFrame:
        df = data.__copy__()
        df = df.rename(self.rename_columns).drop(self.drop_columns)
        data = df.with_columns(*self._convert_to_category())
        return data

    def _convert_to_category(self) -> T.Generator[pl.Expr, T.Any, None]:
        columns = self.categorical_columns

        for column in columns:
            yield pl.col(column).cast(pl.Categorical)
