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

    conf_type = list(dict(object_).keys())

    if conf_type == ["job"]:
        setting = settings.JobSettings.model_validate(object_)
        with setting.job as job:
            job.run()
            return 0

    if conf_type == ["service"]:
        setting = settings.ServiceSettings.model_validate(object_)
        setting.service.start()
        setting.service.stop()
        return 0
