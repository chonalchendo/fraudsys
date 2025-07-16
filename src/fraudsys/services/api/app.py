import polars as pl
from fastapi import Depends, FastAPI

from fraudsys.core import features, schemas
from fraudsys.io import kafka
from fraudsys.services.api import dependencies as deps
from fraudsys.services.api import models as api_models

app = FastAPI()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=api_models.InferenceResponse)
async def submit_transaction(
    trxn: api_models.RawTransaction,
    producer: kafka.KafkaProducerWrapper = Depends(deps.get_kafka_producer),
    preds_producer: kafka.KafkaProducerWrapper = Depends(deps.get_predictions_producer),
) -> api_models.InferenceResponse:
    trxn_message = trxn.model_dump()

    # send to kafka to be consumed by other services
    producer.send(message=trxn_message)

    data = pl.DataFrame(trxn_message)
    cleaned_input = features.clean(data=data)

    # prediction requires a pandas dataframe
    input_ = cleaned_input.to_pandas()
    input = schemas.InputsSchema.check(input_)

    model = deps.get_model()
    output = model.predict(inputs=input)
    pred = output["prediction"].iloc[0]

    # send prediction to kafka topic for monitoring etc.
    pred_message = {
        "transaction_id": trxn_message["trans_num"],
        "prediction": int(pred),
    }
    preds_producer.send(message=pred_message)

    return pred_message
