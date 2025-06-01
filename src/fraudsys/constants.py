TARGET_COLUMN: str = "is_fraud"

SENSITIVE_COLUMNS: list[str] = [
    "first",
    "last",
    "cc_num",
    "dob",
]

CATEGORICAL_COLUMNS: list[str] = ["category"]

NUMERICAL_COLUMNS: list[str] = [
    "amount_usd",
    "lat",
    "long",
    "merch_lat",
    "merch_long",
    "city_pop",
]

RENAME_COLUMNS: dict[str, str] = {
    "trans_date_trans_time": "transaction_time",
    "merchant": "merchant_name",
    "amt": "amount_usd",
    "trans_num": "transaction_id",
}

DROP_COLUMNS: list[str] = [
    "first",
    "last",
    "cc_num",
]
