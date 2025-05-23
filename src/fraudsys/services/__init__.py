from .api.main import APIService
from .cleaner import CleanerService
from .producer import ProducerService

ServiceKind = CleanerService | ProducerService | APIService

__all__ = ["CleanerService", "ProducerService", "APIService"]
