import pydantic as pdt
import pydantic_settings as pdts

from fraudsys import jobs, services


class Settings(pdts.BaseSettings, strict=True, frozen=True, extra="forbid"):
    """Base class for settings."""

    pass


class ServiceSettings(Settings):
    """Service settings for the project.

    Args:
        service (services.ServiceKind): The service to run.
    """

    service: services.ServiceKind = pdt.Field(..., discriminator="KIND")


class JobSettings(Settings):
    """Job settings for the project.

    Args:
        job (jobs.JobKind): The job to run.
    """

    job: jobs.JobKind = pdt.Field(..., discriminator="KIND")
