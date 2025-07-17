import typing as T

import polars as pl

from fraudsys.features.engineering import cleaning
from fraudsys.infra import kafka, logging
from fraudsys.services import base


class FeatureService(base.Service):
    KIND: T.Literal["feature"] = "feature"

    kafka_producer: kafka.KafkaProducerWrapper
    kafka_consumer: kafka.KafkaConsumerWrapper

    logger: logging.Logger = logging.Logger()

    @T.override
    def start(self) -> None:
        self.logger.start()
        logger = self.logger.logger()

        for message in self.kafka_consumer:
            cleaned_message = self._process_message(message.value)
            self.kafka_producer.send(message=cleaned_message)
            logger.debug("CLEANED MESSAGE: {}", cleaned_message)

        self.kafka_producer.get_producer.flush()

    def _process_message(self, message: dict) -> dict[str, T.Any]:
        data = pl.from_records([message])
        transformed_data = cleaning.clean(data)
        dicts = transformed_data.to_dicts()
        if len(dicts) != 1:
            raise ValueError(f"Expected 1 row, got {len(dicts)}")
        return dicts[0]
