import json
import time
import typing as T

import kafka
import pydantic as pdt

from fraudsys import logging


class KafkaProducerWrapper(pdt.BaseModel):
    topic: str
    servers: list[str]

    def send(self, message: dict[str, T.Any]) -> None:
        producer = self._init_producer()
        producer.send(self.topic, message)

    def _init_producer(self) -> kafka.KafkaProducer:
        return kafka.KafkaProducer(
            bootstrap_servers=self.servers,
            value_serializer=lambda v: json.dumps(v).encode(),
        )

    @property
    def get_producer(self) -> kafka.KafkaProducer:
        return self._init_producer()


class KafkaConsumerWrapper(pdt.BaseModel):
    topic: str
    servers: list[str]
    group_id: str

    logger: logging.Logger = logging.Logger()

    def __iter__(self) -> kafka.KafkaConsumer:
        self._consumer = self._create_consumer()
        return self._consumer

    def _create_consumer(self) -> kafka.KafkaConsumer:
        logger = self.logger.logger()
        for i in range(20):
            try:
                consumer = kafka.KafkaConsumer(
                    self.topic,
                    bootstrap_servers=self.servers,
                    group_id=self.group_id,
                    value_deserializer=lambda v: json.loads(v.decode("utf-8")),
                )
                logger.success(
                    "SUCCESS: Initiated Kafka conusmer for topic: {}", self.topic
                )
                return consumer
            except Exception as e:
                logger.exception(
                    f"Trying to instantiate Kafka consumer with bootstrap servers {self.servers} with error {e}"
                )
                time.sleep(10)
                pass
