"""Module for logging configuration and class-level logging decorators."""

import logging
import sys
from collections.abc import Callable
from functools import wraps

# Standard logging setup
logger = logging.getLogger(__name__)


def setup_logging(
    log_level: int = logging.INFO, log_file: str | None = None
) -> logging.Logger:
    """
    Configures the root logger with console and optional file handlers.

    Sets up a standardized format and ensures that existing handlers are cleared
    to prevent duplicate log entries.

    Args:
        log_level (int): The logging level (e.g., logging.DEBUG, logging.INFO).
        log_file (str, optional): Path to a file to save logs. If None,
            logs are only output to stdout.

    Returns:
        logging.Logger: The configured root logger instance.
    """
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Clean up existing handlers to avoid duplicate logs in interactive sessions
    while root_logger.handlers:
        root_logger.removeHandler(root_logger.handlers[0])

    # Console output
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File output if specified
    if log_file:
        try:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
            logger.info(f"File logging initialized at: {log_file}")
        except Exception as e:
            logger.error(
                f"Failed to initialize file handler at {log_file}: {e}"
            )

    return root_logger


def logged(cls: type | None = None, *, name: str = "") -> type | Callable:
    """
    A class decorator that injects a logger and logging methods into a class.

    This decorator wraps the class __init__ method to attach a logger instance
    and provides shortcut methods (self.info, self.error, etc.) directly
    on the class instance.

    Args:
        cls (Type, optional): The class to be decorated.
        name (str, optional): Custom name for the logger. If not provided,
            uses the class name.

    Returns:
        Union[Type, Callable]: The decorated class or a wrapper function
            if a custom name is used.
    """

    def logged_for_init(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            logger_name = name or self.__class__.__name__
            # Attach the logger object
            self.log = logging.getLogger(logger_name)

            # Map standard logging methods to the instance
            for method_name in (
                "debug",
                "info",
                "warning",
                "error",
                "critical",
                "exception",
            ):
                method = getattr(self.log, method_name)
                setattr(self, method_name, method)

            return func(self, *args, **kwargs)

        return wrapper

    def wrap(target_cls: type) -> type:
        target_cls.__init__ = logged_for_init(target_cls.__init__)
        return target_cls

    # Handle both @logged and @logged(name="custom") syntax
    if cls is None:
        return wrap
    return wrap(cls)
