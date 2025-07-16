from functools import lru_cache

import mlflow

from fraudsys.io import kafka, registries
from fraudsys.services.api import models as api_models

_context: api_models.AppContext | None = None


def inject_context(ctx: api_models.AppContext):
    global _context
    _context = ctx


def get_context() -> api_models.AppContext:
    if not _context:
        raise ValueError("_context is None")
    return _context


@lru_cache
def get_kafka_producer() -> kafka.KafkaProducerWrapper:
    ctx: api_models.AppContext = get_context()
    return kafka.KafkaProducerWrapper(
        topic=ctx.raw_transactions_topic, servers=ctx.kafka_servers
    )


@lru_cache
def get_predictions_producer() -> kafka.KafkaProducerWrapper:
    ctx: api_models.AppContext = get_context()
    return kafka.KafkaProducerWrapper(
        topic=ctx.predictions_topic, servers=ctx.kafka_servers
    )


@lru_cache
def get_model() -> registries.CustomReader.Adapter:
    ctx: api_models.AppContext = get_context()
    logger = ctx.logger.logger()
    logger.info(
        "Reading model",
        tracking_server=ctx.mlflow_tracking_uri,
        registry=ctx.mlflow_registry,
        alias=ctx.mlflow_model_alias,
    )
    # Set the tracking and registry URIs
    mlflow.set_tracking_uri(ctx.mlflow_tracking_uri)
    mlflow.set_registry_uri(ctx.mlflow_tracking_uri)  # same as tracking for local

    model_uri = registries.uri_for_model_alias_or_version(
        name=ctx.mlflow_registry, alias_or_version=ctx.mlflow_model_alias
    )
    model = registries.CustomReader().read(uri=model_uri)
    return model
