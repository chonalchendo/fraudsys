from .evaluation import EvaluationJob
from .experiment import ExperimentJob
from .extract import ExtractJob
from .inference import InferenceJob
from .offline_features import OfflineFeaturesJob
from .promotion import PromotionJob
from .training import TrainingJob
from .tuning import TuningJob
from .explanation import ExplanationsJob

JobKind = (
    ExtractJob
    | ExperimentJob
    | OfflineFeaturesJob
    | TuningJob
    | TrainingJob
    | PromotionJob
    | InferenceJob
    | EvaluationJob
    | ExplanationsJob
)
