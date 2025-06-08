import argparse
from pathlib import Path
import typing as T

import omegaconf as oc
from dotenv import load_dotenv
from rich import print

from fraudsys import settings
from fraudsys.io import configs

# load env variables
load_dotenv()

JOBS_DIR = Path("confs/jobs")
SERVICES_DIR = Path("confs/services")


def get_config_options(
    config_type: T.Literal["jobs", "services"], format: str = "yaml"
) -> list[str]:
    """Function to load configuration files for jobs or services."""
    directory = f"confs/{config_type}"
    return [
        str(file).split("/")[-1].replace(f".{format}", "")
        for file in list(Path(directory).glob(f"*.{format}"))
    ]


parser = argparse.ArgumentParser(
    description="Run a job or service from YAML/JSON configuration files or by name."
)

# Create subparsers for different commands
subparsers = parser.add_subparsers(dest="command", help="Available commands")


# Named job execution
job_parser = subparsers.add_parser("job", help="Run a named job")
job_parser.add_argument("name", choices=get_config_options("jobs"), help="Job name")
job_parser.add_argument(
    "-e", "--extras", nargs="*", default=[], help="Config strings for the job."
)
job_parser.add_argument("-c", "--config", help="Override config file path")

# Named service execution
service_parser = subparsers.add_parser("service", help="Run a named service")
service_parser.add_argument(
    "name", choices=get_config_options("services"), help="Service name"
)
service_parser.add_argument(
    "-e", "--extras", nargs="*", default=[], help="Config strings for the job."
)
service_parser.add_argument("-c", "--config", help="Override config file path")

# List commands
list_parser = subparsers.add_parser("list", help="List available jobs/services")
list_parser.add_argument("type", choices=["jobs", "services"], help="What to list")


def execute(argv: list[str] | None = None) -> int:
    """Execute the CLI with the given arguments."""
    args = parser.parse_args(argv)

    # Handle list command
    if args.command == "list":
        if args.type == "jobs":
            print("Available jobs:")
            for job in get_config_options("jobs"):
                print(f"  - {job}")
        elif args.type == "services":
            print("Available services:")
            for service in get_config_options("services"):
                print(f"  - {service}")
        return 0

    if args.command == "job":
        config_file = JOBS_DIR / f"{args.name}.yaml"
        config_files = [config_file]
    elif args.command == "service":
        config_file = SERVICES_DIR / f"{args.name}.yaml"
        config_files = [config_file]
    else:
        # Fallback to original behavior for backward compatibility
        if not args.files:
            raise RuntimeError("No configs provided.")
        config_files = args.files

    # Parse configs (your original logic)
    files = [configs.parse_file(file) for file in config_files]
    strings = [configs.parse_string(string) for string in getattr(args, "extras", [])]

    config = configs.merge_configs([*files, *strings])

    if not isinstance(config, oc.DictConfig):
        raise RuntimeError("Config is not a dictionary")

    object_ = configs.to_object(config)

    if not isinstance(object_, dict):
        raise RuntimeError("Expected object_ to be a dict")

    if args.command == "job":
        job_setting = settings.JobSettings.model_validate(object_)
        print(job_setting)
        with job_setting.job as job:
            job.run()
            return 0

    if args.command == "service":
        service_setting = settings.ServiceSettings.model_validate(object_)
        print(service_setting)
        service_setting.service.start()
        service_setting.service.stop()
        return 0
