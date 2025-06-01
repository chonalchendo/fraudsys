from .experiment import ExperimentJob
from .extract import ExtractJob
from .offline_features import OfflineFeaturesJob

JobKind = ExtractJob | ExperimentJob | OfflineFeaturesJob
