import pydantic as pdt

from fraudsys.infra import logging


class AppContext(pdt.BaseModel):
    kafka_servers: list[str]
    raw_transactions_topic: str
    predictions_topic: str
    mlflow_tracking_uri: str = "http://mlflow:5000"
    mlflow_registry: str = "fraudsys"
    mlflow_model_alias: str = "Champion"
    logger: logging.Logger


class RawTransaction(pdt.BaseModel):
    customer_id: str
    trans_num: str
    trans_date_trans_time: str
    cc_num: int
    merchant: str
    category: str
    amt: float
    first: str
    last: str
    gender: str
    street: str
    city: str
    state: str
    zip: int
    lat: float
    long: float
    city_pop: int
    job: str
    dob: str
    unix_time: int
    merch_lat: float
    merch_long: float


class InferenceResponse(pdt.BaseModel):
    transaction_id: str
    prediction: int
