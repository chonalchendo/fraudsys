from functools import lru_cache

from fastapi import Depends, FastAPI

from fraudsys.io import kafka
from fraudsys.services.api import models

_context: models.AppContext | None = None


def inject_context(ctx: models.AppContext):
    global _context
    _context = ctx


def get_context() -> models.AppContext:
    if not _context:
        raise ValueError('_context is None')
    return _context


@lru_cache
def get_kafka_producer() -> kafka.KafkaProducerWrapper:
    ctx: models.AppContext = get_context()
    return kafka.KafkaProducerWrapper(topic=ctx.input_topic, servers=ctx.kafka_servers)


app = FastAPI()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/transaction")
async def submit_transaction(
    trxn: models.RawTransaction,
    producer: kafka.KafkaProducerWrapper = Depends(get_kafka_producer),
) -> dict[str, str]:
    trxn_message = trxn.model_dump()
    producer.send(message=trxn_message)
    return {"transaction_id": trxn_message["trans_num"]}
