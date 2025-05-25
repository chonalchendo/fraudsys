import argparse

import omegaconf as oc
from dotenv import load_dotenv

from fraudsys import settings
from fraudsys.io import configs

# load env variables
load_dotenv()


parser = argparse.ArgumentParser(
    description="Run a job or service from YAML/JSON configuration files."
)
parser.add_argument("files", nargs="*", help="Config files for the job")
parser.add_argument(
    "-e", "--extras", nargs="*", default=[], help="Config strings for the job."
)


def execute(argv: list[str] | None = None) -> int:
    """Execute the CLI with the given arguments."""
    args = parser.parse_args(argv)

    files = [configs.parse_file(file) for file in args.files]
    strings = [configs.parse_string(string) for string in args.extras]

    if len(files) == 0:
        raise RuntimeError("No configs provided.")

    config = configs.merge_configs([*files, *strings])

    if not isinstance(config, oc.DictConfig):
        raise RuntimeError("Config is not a dictionary")

    object_ = configs.to_object(config)

    if not isinstance(object_, dict):
        raise RuntimeError("Expected object_ to be a dict")

    conf_type = list(object_.keys())

    if conf_type == ["job"]:
        job_setting = settings.JobSettings.model_validate(object_)
        with job_setting.job as job:
            job.run()
            return 0

    if conf_type == ["service"]:
        service_setting = settings.ServiceSettings.model_validate(object_)
        service_setting.service.start()
        service_setting.service.stop()
        return 0

    raise RuntimeError(f"Unsupported config type: {conf_type}")
