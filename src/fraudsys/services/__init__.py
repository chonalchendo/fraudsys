from .cleaner import CleanerService
from .producer import ProducerService

ServiceKind = CleanerService | ProducerService

__all__ = ["CleanerService", "ProducerService"]
