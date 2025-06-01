import typing as T

from imblearn import over_sampling, pipeline
from sklearn import compose, preprocessing

if T.TYPE_CHECKING:
    from fraudsys.core import models


def create_pipeline(
    model: "models.ModelKind",
    numerical_columns: list[str],
    category_columns: list[str],
    random_state: int,
) -> pipeline.Pipeline:
    # subcomponents
    categoricals_transformer = preprocessing.OneHotEncoder(
        sparse_output=False, handle_unknown="ignore"
    )
    # components
    transformer = compose.ColumnTransformer(
        [
            ("categoricals", categoricals_transformer, category_columns),
            ("numericals", "passthrough", numerical_columns),
        ],
        remainder="drop",
    )
    # pipeline
    return pipeline.Pipeline(
        steps=[
            ("transformer", transformer),
            ("smote", over_sampling.SMOTE(random_state=random_state)),
            ("model", model),
        ]
    )
