from __future__ import annotations

import sys
import typing as T

import loguru
import pydantic as pdt


class Logger(pdt.BaseModel, strict=True, frozen=True, extra="forbid"):
    """Project logger.

    Args:
        sink (str): Sink to log messages. Defaults to "stderr".
        level (str): Level to log messages. Defaults to "DEBUG".
        format (str): Format of the log message. Defaults to:
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green>"
            "<level>{level: <8}</level>"
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan>"
            " <level>{message}</level>"
        serialize (bool): Serialize the log message. Defaults to False.
        colorize (bool): Colorize the log message. Defaults to True.
        backtrace (bool): Show backtrace. Defaults to True.
        diagnose (bool): Diagnose the log message. Defaults to False.
        catch (bool): Catch exceptions. Defaults to True.
    """

    sink: str = "stderr"
    level: str = "DEBUG"
    format: str = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<magenta>{function}</magenta>:<yellow>{line}</yellow> | "
        "<level>{message}</level>"
    )
    serialize: bool = False
    colorize: bool = True
    backtrace: bool = True
    diagnose: bool = False
    catch: bool = True

    @T.override
    def start(self) -> None:
        loguru.logger.remove()
        config = self.model_dump()

        sink_mapping = {
            "stderr": sys.stderr,
            "sys.stderr": sys.stderr,
            "stdout": sys.stdout,
            "sys.stdout": sys.stdout,
        }
        config["sink"] = sink_mapping.get(config["sink"], config["sink"])
        loguru.logger.add(**config)

    def stop(self) -> None:
        pass

    def logger(self) -> loguru.Logger:
        """Return the logger.

        Returns:
            loguru.Logger: logger object.
        """
        return loguru.logger
