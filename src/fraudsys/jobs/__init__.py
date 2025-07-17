from .data.extract import ExtractJob
from .data.offline_feature_engineering import OfflineFeatureEngineeringJob
from .data.offline_features import OfflineFeaturesJob
from .ml.evaluation import EvaluationJob
from .ml.experiment import ExperimentJob
from .ml.explanation import ExplanationsJob
from .ml.inference import InferenceJob
from .ml.promotion import PromotionJob
from .ml.training import TrainingJob
from .ml.tuning import TuningJob

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
