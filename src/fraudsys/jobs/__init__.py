from .experiment import ExperimentJob
from .extract import ExtractJob
from .offline_features import OfflineFeaturesJob
from .tuning import TuningJob

JobKind = ExtractJob | ExperimentJob | OfflineFeaturesJob | TuningJob
