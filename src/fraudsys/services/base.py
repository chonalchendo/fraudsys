import abc

import pydantic as pdt


class Service(abc.ABC, pdt.BaseModel, strict=True, frozen=True, extra="forbid"):
    """Base class for all services."""

    @abc.abstractmethod
    def start(self) -> None:
        """Start the service."""
        pass

    def stop(self) -> None:
        """Stop the service."""
        pass
