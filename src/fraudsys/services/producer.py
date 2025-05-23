import json
import typing as T
from time import sleep

import pydantic as pdt
from kafka import KafkaAdminClient, KafkaProducer
from kafka.admin import NewTopic

from fraudsys import logging as logger_
from fraudsys.io import datasets
from fraudsys.services import base


class ProducerService(base.Service):
    KIND: T.Literal["producer"] = "producer"

    topic: str
    servers: list[str]

    input: datasets.LoaderKind = pdt.Field(..., discriminator="KIND")

    logger: logger_.Logger = logger_.Logger()

    @T.override
    def start(self) -> None:
        self.logger.start()
        logger = self.logger.logger()

        producer, admin = self._initialise()
        self._create_topic(admin)

        data = self.input.load()

        for record in data.to_dicts():
            producer.send(self.topic, json.dumps(record).encode())
            logger.debug("MESSAGE: {}", record)
            sleep(1)

    @T.override
    def stop(self) -> None:
        logger = self.logger.logger()
        try:
            admin = KafkaAdminClient(bootstrap_servers=self.servers)
            logger.info(admin.delete_topics([self.topic]))
            logger.info(f"Topic {self.topic} deleted")
        except Exception as e:
            logger.exception(str(e))
            pass

    def _initialise(self) -> tuple[KafkaProducer, KafkaAdminClient]:
        logger = self.logger.logger()
        for i in range(20):
            try:
                producer = KafkaProducer(bootstrap_servers=self.servers)
                admin = KafkaAdminClient(bootstrap_servers=self.servers)
                logger.success("SUCCESS: instantiated Kafka admin and producer")
                return producer, admin
            except Exception as e:
                logger.exception(
                    f"Trying to instantiate admin and producer with bootstrap servers {self.servers} with error {e}"
                )
                sleep(10)
                pass

    def _create_topic(self, admin: KafkaAdminClient):
        logger = self.logger.logger()
        try:
            topic = NewTopic(name=self.topic, num_partitions=3, replication_factor=1)
            admin.create_topics([topic])
            logger.info(f"Topic {self.topic} created")
        except Exception as e:
            logger.exception(str(e))
            pass
