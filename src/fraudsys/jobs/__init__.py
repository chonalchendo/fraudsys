from .experiment import ExperimentJob
from .extract import ExtractJob
from .offline_features import OfflineFeaturesJob
from .training import TrainingJob
from .tuning import TuningJob

JobKind = ExtractJob | ExperimentJob | OfflineFeaturesJob | TuningJob | TrainingJob
