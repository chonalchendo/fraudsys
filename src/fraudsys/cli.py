from __future__ import annotations

import typing as T
from enum import Enum
from pathlib import Path

import cyclopts
import omegaconf as oc
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from fraudsys import configs, settings

# Load environment variables
load_dotenv()

# Initialize Cyclopts app
app = cyclopts.App(
    name="fraudsys",
    help="FraudSys CLI - Run jobs and services from configuration files",
)

console = Console()

# Constants
CONFS_DIR = Path("confs")
JOBS_DIR = CONFS_DIR / "jobs"
SERVICES_DIR = CONFS_DIR / "services"


class ConfigType(str, Enum):
    """Configuration types available."""

    JOBS = "jobs"
    SERVICES = "services"


class JobType(str, Enum):
    """Job subtypes available."""

    DATA = "data"
    ML = "ml"


def get_available_configs(
    config_type: ConfigType,
    sub_type: JobType | None = None,
    extension: str = "yaml",
) -> T.Sequence[str]:
    """Get list of available configuration files."""
    if config_type == ConfigType.JOBS and sub_type:
        directory = JOBS_DIR / sub_type.value
    elif config_type == ConfigType.JOBS:
        # Get all jobs from both data and ml subdirectories
        data_jobs = list((JOBS_DIR / JobType.DATA.value).glob(f"*.{extension}"))
        ml_jobs = list((JOBS_DIR / JobType.ML.value).glob(f"*.{extension}"))
        return [f.stem for f in data_jobs + ml_jobs]
    else:
        directory = CONFS_DIR / config_type.value

    if not directory.exists():
        return []

    return [f.stem for f in directory.glob(f"*.{extension}")]


def find_job_config(job_name: str) -> Path:
    """Find the configuration file for a given job name."""
    # Check in data jobs
    data_config = JOBS_DIR / JobType.DATA.value / f"{job_name}.yaml"
    if data_config.exists():
        return data_config

    # Check in ml jobs
    ml_config = JOBS_DIR / JobType.ML.value / f"{job_name}.yaml"
    if ml_config.exists():
        return ml_config

    raise FileNotFoundError(f"Job configuration not found: {job_name}")


def load_and_merge_configs(
    config_files: T.Sequence[Path],
    extras: T.Sequence[str] | None = None,
) -> oc.DictConfig:
    """Load configuration files and merge with extra parameters."""
    # Parse config files
    file_configs = [configs.parse_file(str(file)) for file in config_files]

    # Parse extra strings
    string_configs = []
    if extras:
        string_configs = [configs.parse_string(string) for string in extras]

    # Merge all configs
    config = configs.merge_configs([*file_configs, *string_configs])

    if not isinstance(config, oc.DictConfig):
        raise RuntimeError("Config is not a dictionary")

    return config


@app.command
def job(
    name: str,
    *,
    extras: T.Annotated[
        T.Sequence[str] | None,
        cyclopts.Parameter(help="Additional config overrides in key=value format"),
    ] = None,
    config: T.Annotated[
        Path | None,
        cyclopts.Parameter(help="Override config file path"),
    ] = None,
) -> None:
    """Run a job by name.

    Examples:
        fraudsys job extract_features
        fraudsys job train_model --extras "batch_size=64" "epochs=10"
        fraudsys job evaluate --config custom_config.yaml
    """
    try:
        # Determine config file
        if config:
            config_file = config
        else:
            config_file = find_job_config(name)

        console.print(f"[blue]Running job:[/blue] {name}")
        console.print(f"[dim]Config:[/dim] {config_file}")

        # Load and merge configs
        merged_config = load_and_merge_configs([config_file], extras)

        # Convert to object and validate
        object_ = configs.to_object(merged_config)

        if not isinstance(object_, dict):
            raise RuntimeError("Expected object_ to be a dict")

        # Create and run job
        job_settings = settings.JobSettings.model_validate(object_)

        with job_settings.job as job:
            console.print(f"[green]✓[/green] Job started: {job.__class__.__name__}")
            job.run()
            console.print(f"[green]✓[/green] Job completed successfully")

    except FileNotFoundError as e:
        console.print(f"[red]✗ Error:[/red] {e}")
        console.print("\n[dim]Available jobs:[/dim]")
        list_jobs()
        raise cyclopts.ValidationError(str(e))
    except Exception as e:
        console.print(f"[red]✗ Error:[/red] {e}")
        raise cyclopts.ValidationError(str(e))


@app.command
def service(
    name: str,
    *,
    extras: T.Annotated[
        T.Sequence[str] | None,
        cyclopts.Parameter(help="Additional config overrides in key=value format"),
    ] = None,
    config: T.Annotated[
        Path | None,
        cyclopts.Parameter(help="Override config file path"),
    ] = None,
) -> None:
    """Run a service by name.

    Examples:
        fraudsys service api
        fraudsys service monitoring --extras "port=8080"
        fraudsys service producer --config custom_service.yaml
    """
    try:
        # Determine config file
        if config:
            config_file = config
        else:
            config_file = SERVICES_DIR / f"{name}.yaml"
            if not config_file.exists():
                raise FileNotFoundError(f"Service configuration not found: {name}")

        console.print(f"[blue]Starting service:[/blue] {name}")
        console.print(f"[dim]Config:[/dim] {config_file}")

        # Load and merge configs
        merged_config = load_and_merge_configs([config_file], extras)

        # Convert to object and validate
        object_ = configs.to_object(merged_config)

        if not isinstance(object_, dict):
            raise RuntimeError("Expected object_ to be a dict")

        # Create and run service
        service_settings = settings.ServiceSettings.model_validate(object_)

        console.print(
            f"[green]✓[/green] Service initialized: {service_settings.service.__class__.__name__}"
        )
        service_settings.service.start()
        console.print(f"[green]✓[/green] Service started")

        # Note: In production, you might want to keep the service running
        # instead of immediately stopping it
        service_settings.service.stop()
        console.print(f"[yellow]![/yellow] Service stopped")

    except FileNotFoundError as e:
        console.print(f"[red]✗ Error:[/red] {e}")
        console.print("\n[dim]Available services:[/dim]")
        list_services()
        raise cyclopts.ValidationError(str(e))
    except Exception as e:
        console.print(f"[red]✗ Error:[/red] {e}")
        raise cyclopts.CycloptsError(str(e))


@app.command
def list(
    config_type: T.Annotated[
        ConfigType,
        cyclopts.Parameter(help="Type of configurations to list"),
    ],
) -> None:
    """List available jobs or services.

    Examples:
        fraudsys list jobs
        fraudsys list services
    """
    if config_type == ConfigType.JOBS:
        list_jobs()
    elif config_type == ConfigType.SERVICES:
        list_services()


def list_jobs() -> None:
    """List all available jobs organized by type."""
    table = Table(title="Available Jobs", show_header=True)
    table.add_column("Type", style="cyan")
    table.add_column("Name", style="green")
    table.add_column("Config Path", style="dim")

    # Data jobs
    data_jobs = get_available_configs(ConfigType.JOBS, JobType.DATA)
    for job in sorted(data_jobs):
        table.add_row("data", job, f"confs/jobs/data/{job}.yaml")

    # ML jobs
    ml_jobs = get_available_configs(ConfigType.JOBS, JobType.ML)
    for job in sorted(ml_jobs):
        table.add_row("ml", job, f"confs/jobs/ml/{job}.yaml")

    console.print(table)
    console.print(
        f"\n[dim]Total:[/dim] {len(data_jobs)} data jobs, {len(ml_jobs)} ML jobs"
    )


def list_services() -> None:
    """List all available services."""
    table = Table(title="Available Services", show_header=True)
    table.add_column("Name", style="green")
    table.add_column("Config Path", style="dim")

    services = get_available_configs(ConfigType.SERVICES)
    for service in sorted(services):
        table.add_row(service, f"confs/services/{service}.yaml")

    console.print(table)
    console.print(f"\n[dim]Total:[/dim] {len(services)} services")


@app.command
def validate(
    config_path: T.Annotated[
        Path,
        cyclopts.Parameter(help="Path to configuration file to validate"),
    ],
    *,
    type: T.Annotated[
        T.Literal["job", "service"],
        cyclopts.Parameter(help="Type of configuration to validate"),
    ],
) -> None:
    """Validate a configuration file without running it.

    Examples:
        fraudsys validate confs/jobs/ml/train.yaml --type job
        fraudsys validate confs/services/api.yaml --type service
    """
    try:
        console.print(f"[blue]Validating {type} config:[/blue] {config_path}")

        # Load config
        config = configs.parse_file(str(config_path))
        object_ = configs.to_object(config)

        if not isinstance(object_, dict):
            raise RuntimeError("Expected object_ to be a dict")

        # Validate based on type
        if type == "job":
            settings.JobSettings.model_validate(object_)
            console.print("[green]✓[/green] Job configuration is valid")
        else:
            settings.ServiceSettings.model_validate(object_)
            console.print("[green]✓[/green] Service configuration is valid")

    except Exception as e:
        console.print(f"[red]✗ Validation failed:[/red] {e}")
        raise cyclopts.CycloptsError(str(e))


def execute(argv: T.Sequence[str] | None = None) -> int:
    """Execute the CLI with the given arguments."""
    try:
        app(argv)
        return 0
    except cyclopts.ValidationError as e:
        console.print(f"[red]Error:[/red] {e}")
        return 1
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        return 130
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {e}")
        return 1
