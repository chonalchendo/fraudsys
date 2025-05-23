import json
import typing as T

import pydantic as pdt
from kafka import KafkaProducer


class KafkaProducerWrapper(pdt.BaseModel):
    topic: str
    servers: list[str]

    def send(self, message: dict[str, T.Any]) -> None:
        producer = self._get_producer()
        producer.send(self.topic, message)

    def _get_producer(self) -> KafkaProducer:
        return KafkaProducer(
            bootstrap_servers=self.servers,
            value_serializer=lambda v: json.dumps(v).encode(),
        )
