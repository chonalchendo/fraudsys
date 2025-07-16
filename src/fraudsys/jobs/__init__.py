from .evaluation import EvaluationJob
from .experiment import ExperimentJob
from .explanation import ExplanationsJob
from .extract import ExtractJob
from .inference import InferenceJob
from .offline_feature_engineering import OfflineFeatureEngineeringJob
from .offline_features import OfflineFeaturesJob
from .promotion import PromotionJob
from .training import TrainingJob
from .tuning import TuningJob

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
    | OfflineFeatureEngineeringJob
)
