import time
import typing as T
from collections import deque

from prometheus_client import Counter, Gauge, Histogram, Info, start_http_server

from fraudsys.io import kafka, runtimes
from fraudsys.services import base

# Prometheus metrics
fraud_predictions_total = Counter(
    "fraud_predictions_total", "Total predictions", ["prediction"]
)
prediction_confidence = Histogram(
    "prediction_confidence", "Prediction confidence scores"
)
fraud_rate_gauge = Gauge(
    "fraud_rate_current", "Current fraud rate (last 1000 predictions)"
)
predictions_per_minute = Gauge("predictions_per_minute", "Predictions per minute")
service_info = Info("monitoring_service", "Monitoring service information")


class MonitoringService(base.Service):
    KIND: T.Literal["monitoring"] = "monitoring"

    port: int = 8001
    prediction_consumer: kafka.KafkaConsumerWrapper

    recent_predictions: deque = deque(maxlen=1000)
    prediction_timestamps: deque = deque(maxlen=1000)

    logger: runtimes.Logger = runtimes.Logger()

    @T.override
    def start(self) -> None:
        self.logger.start()
        logger = self.logger.logger()

        service_info.info({"status": "running"})

        try:
            start_http_server(self.port)
            logger.info(f"Prometheus metrics server started on port {self.port}")
        except OSError as e:
            logger.error(f"Failed to start metrics server on port {self.port}: {e}")
            raise

        try:
            for message in self.prediction_consumer:
                try:
                    data: dict = message.value
                    logger.debug(f"Prediction: {data}")
                    self._process_prediction(data)
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    continue
        except Exception as e:
            logger.error(f"Kafka consumer error: {e}")
            raise

    def _process_prediction(self, data: dict) -> None:
        logger = self.logger.logger()
        prediction = data.get("prediction")
        if prediction is None:
            logger.warning("Received message without prediction field")
            return

        # Update metrics
        fraud_predictions_total.labels(prediction=str(prediction)).inc()

        # Track recent predictions for fraud rate
        self.recent_predictions.append(prediction)
        self.prediction_timestamps.append(time.time())

        # Update gauges
        if len(self.recent_predictions) > 0:
            fraud_rate_gauge.set(
                sum(self.recent_predictions) / len(self.recent_predictions)
            )

        # Calculate predictions per minute
        current_time = time.time()
        recent_count = sum(
            1 for ts in self.prediction_timestamps if current_time - ts <= 60
        )
        predictions_per_minute.set(recent_count)
