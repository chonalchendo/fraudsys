import pydantic as pdt


class AppContext(pdt.BaseModel):
    kafka_servers: list[str]
    input_topic: str


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
