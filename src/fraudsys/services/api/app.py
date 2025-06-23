from functools import lru_cache

import mlflow
import polars as pl
from fastapi import Depends, FastAPI

from fraudsys.core import features, schemas
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
    return kafka.KafkaProducerWrapper(topic=ctx.input_topic, servers=ctx.kafka_servers)


@lru_cache
def get_model() -> registries.CustomReader.Adapter:
    ctx: api_models.AppContext = get_context()

    # Set the tracking and registry URIs
    mlflow.set_tracking_uri(ctx.mlflow_tracking_uri)
    mlflow.set_registry_uri(ctx.mlflow_tracking_uri)  # same as tracking for local

    model_uri = registries.uri_for_model_alias_or_version(
        name=ctx.mlflow_registry, alias_or_version=ctx.mlflow_model_alias
    )
    print(model_uri)
    model = registries.CustomReader().read(uri=model_uri)
    return model


app = FastAPI()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
async def submit_transaction(
    trxn: api_models.RawTransaction,
    producer: kafka.KafkaProducerWrapper = Depends(get_kafka_producer),
):
    trxn_message = trxn.model_dump()

    # send to kafka to be consumed by other services
    producer.send(message=trxn_message)

    data = pl.DataFrame(trxn_message)
    cleaned_input = features.clean(data=data)

    # prediction requires a pandas dataframe
    input_ = cleaned_input.to_pandas()
    input = schemas.InputsSchema.check(input_)

    model = get_model()
    output = model.predict(inputs=input)
    pred = output['prediction'].iloc[0]

    # send prediction to kafka topic for monitoring etc.

    return {"transaction_id": trxn_message["trans_num"], "prediction": int(pred)} 
